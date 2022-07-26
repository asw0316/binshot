import os, sys
import logging
import re
import json
from collections import Counter
from capstone import *

import unit
import util
import op
import glibc

linker_funcs = ['__libc_csu_init', '__libc_csu_fini', 'deregister_tm_clones', '_start',
                 'register_tm_clones', '__do_global_dtors_aux', 'frame_dummy']

class Normalization():
    """
    Instruction Normalization class
    """

    def __init__(self, dumped_funcs):
        '''
        Input as a series of dumped functions from IDA
        '''
        self.dumped_funcs = dumped_funcs
        self.normalized_instns = []
        self.corpus_data = {}   # (func_name: normalized_instructions)
        self.Fs= list()
        self.bin_info = None

    def get_normalized_ins_ctr(self):
        return Counter(self.normalized_instns)

    def _copy_instr_over(self, I):
        """
        Copy an IDA_Instruction instance into Instruction
        :param I:
        :return:
        """
        UI = unit.Instruction(I.start, I.end)
        UI.sz = I.sz
        UI.is_call = I.is_call

        UI.parent = I.parent
        UI.callee = I.callee
        UI.raw_bytes = I.raw_bytes
        UI.ref_string = I.ref_string
        UI.glibc_func = I.glibc_func

        return UI

    def _copy_bb_over(self, B):
        """
        Copy an IDA_BasicBlock instance into BasicBlock
        :param B:
        :return:
        """
        UB = unit.BasicBlock(B.start, B.end)
        UB.sz = B.sz
        UB.parent = B.parent
        UB.instns = B.instns
        UB.num_ins = B.num_ins
        UB.is_entry = B.is_entry
        UB.is_exit = B.is_exit
        UB.align_sz = B.align_sz
        return UB

    def _copy_func_over(self, F):
        """
        Copy an IDA_Function instance into Function
        :param F:
        :return:
        """
        UF = unit.Function(F.start, F.end)
        UF.sz = F.sz
        UF.name = F.name
        UF.align_sz = F.align_sz
        UF.demangled = F.demangled
        #UF.no_return = F.no_return
        UF.bbls = F.bbls
        UF.num_bbs = F.num_bbs

        UF.refs_from = F.refs_from
        UF.call_refs_to = F.call_refs_to
        UF.jump_refs_to = F.jump_refs_to
        UF.data_refs_to = F.data_refs_to

        UF.is_recursive = F.is_recursive
        UF.is_linker_func = F.is_linker_func
        UF.align_sz = F.align_sz
        return UF

    # Note that the following function should be always
    # invoked after self.disassemble()
    def disassemble_and_normalize_instructions(self, normalization_level=3):
        """
        Disassemble instructions with the Capstone disassembly engine
            : consider user-defined functions only (i.e., funcs from a linker)
            : remove unnecessary instructions (i.e., nop)
        Normalize each instruction:
            make sure a 'instr' contains a single instruction
        Normalization level
            0: None -> original disassembly
            1: immediate value -> imm (e.g., 0x41 -> imm)
            2: register -> regname_regsize (e.g., rax -> reg8)
            3: pointer -> ptr_ptrsize (e.g., dword -> ptr4)
        :param instr:
        :return:
        """

        # Normalization level == 1 (immediate value)
        def __normalize_immediate(I, opr):
            opcode = I.opcode
            is_addr = lambda x: x.startswith('0x')
            in_section = lambda x, y: self.bin_info.ep.getSectionByVA(x).startswith(y) \
                if self.bin_info.ep.getSectionByVA(x) else False

            # Is an opcode a call invocation for...
            if I.is_call:
                # Glibc function?
                if I.glibc_func is not None:
                    return 'libc' + I.glibc_func 
                # Recursive function?
                elif I.callee > 0 and I.callee == I.parent.parent.start:
                    return 'self'
                # Inner function within a binary?
                elif self.bin_info.ep.getSectionByVA(I.callee) == '.text':
                    return 'innerfunc'
                # External function defined in a library other than glibc?
                elif in_section(I.callee, '.plt') or in_section(I.callee, '.got'):
                    return 'externfunc' 
                # Something else? (Not expected!)
                else:
                    return 'sthelse'

            # Is an opcode a jump?
            elif opcode in op.X86_JMP:
                return 'jmpdst'

            # Is an operand pointing to a string in .rodata?
            elif is_addr(opr) and opcode.startswith('mov') and len(I.ref_string) > 0:
                return 'dispstr'

            # Is an operand pointing to an uninitialized value in .bss section?
            elif is_addr(opr) and in_section(int(opr, 16), '.bss'):
                return 'dispbss'

            # Is an operand pointing to other data in either .rodata or .data section?
            elif is_addr(opr) and (in_section(int(opr, 16), '.data') or in_section(int(opr, 16), '.rodata')):
                return 'dispdata'

            # Otherwise
            else:
                return 'immval'

        # Normalization level == 2 (register)
        def __normalize_register(oper_head):
            return op.X86_REG_OPERANDS[oper_head]

        # Normalization level == 3 (pointer)
        def __normalize_ptr(operand, ref_str_access=False):
            """
            memory address that a pointer points to:
              (base register + index register * scaling factor [+/-] displacement)
            """
            ptr_target = operand[operand.find('[') + 1: operand.find(']')]
            opers = []

            for c in ptr_target:
                if "+" in c:
                    opers.append("+")
                if "-" in c:
                    opers.append("-")
                if "*" in c:
                    opers.append("*")

            ptr_target_components = [x.strip() for x in re.split("\+|\-|\*", ptr_target)]
            for idx, component in enumerate(ptr_target_components):
                if component.startswith('0x'):
                    ptr_target_components[idx] = 'dispstr' if ref_str_access else 'disp'
                elif component in op.X86_REG_OPERANDS:
                    ptr_target_components[idx] = op.X86_REG_OPERANDS[component]
                elif component.isdigit():
                    continue

            converted_ptr_operand = ''
            for i, t in enumerate(ptr_target_components):
                if i < len(opers):
                    converted_ptr_operand += t + opers[i]
                else:
                    converted_ptr_operand += t

            return operand[:operand.find('[')].replace(" ","") + "[" + converted_ptr_operand + "]"

        if not self.bin_info:
            logging.critical("Instructions have not been disassembled yet!")
            return

        md = Cs(CS_ARCH_X86, CS_MODE_64)
        md.detail = True
        failed_disassembly_ctr = 0

        bar = util.ProgressBar(self.bin_info.num_fns,
                               name="Disassemble/Normalize: " + self.bin_info.bin_name)

        # Now we have a new link list at each level to traverse all instances in handy
        F = self.bin_info.first_function
        
        #create list of functions in the binary
        while F:
            self.Fs.append(F.name)
            F = F.next
        F = self.bin_info.first_function

        while F:
            # [a] Mark if a function is not a user-defined one
            if F.name in linker_funcs:
                F.is_linker_func = True
                self.bin_info.num_linker_func += 1
                #F = F.next
                #bar += 1
                #continue

            B = F.first_basic_block

            normalized_instrunctions = ''
            while B:
                I = B.first_instruction
                while I:
                    try:
                        # Get the capstone instruction instance
                        I.cs_instr = [x for x in md.disasm(I.raw_bytes, 0x0)][0]
                        # [b] Mark if an instruction does nothing (i.e., nop)
                        if not I.cs_instr or I.cs_instr.mnemonic == 'nop':
                            I.is_nop = True
                            I.opcode = I.cs_instr.mnemonic
                            I = I.next
                            continue

                    except:
                        import binascii
                        failed_disassembly_ctr += 1

                    try:
                        I.opcode = I.cs_instr.mnemonic
                        I.operands_str = I.cs_instr.op_str
                        opcode = I.opcode
                        operands_str = I.operands_str
                        operands = [x.strip() for x in operands_str.strip().split(',')]

                        if len(operands_str) > 0:
                            for idx, opr in enumerate(I.cs_instr.operands):
                                try:
                                    operand = operands[idx]
                                    orig_val = operand
                                    I.operands.append(operand)
                                    if normalization_level > 0: # immediate value
                                        # We ignore an immediate value as a signature iff
                                        # it represents either a call or jmp target
                                        if opr.access == CS_AC_INVALID:
                                            operands[idx] = __normalize_immediate(I, operands[idx])
                                            if operands[idx] == 'immval':
                                                I.imms.append(orig_val)
                                                I.has_immediate = True

                                    if normalization_level > 1: # registers
                                        oper_head = operand.split(" ")[0].strip()
                                        if oper_head in op.X86_REG_OPERANDS:
                                            operands[idx] = __normalize_register(oper_head)

                                    if normalization_level > 2: # pointers
                                        ref_str_access = True if len(I.ref_string) > 0 else False
                                        if 'ptr' in operand and '[' in operand:
                                            operands[idx] = __normalize_ptr(operand, ref_str_access=ref_str_access)
                                        if 'lea' in opcode:
                                            operands[1] = __normalize_ptr(operands[1], ref_str_access=ref_str_access)

                                except IndexError:
                                    logging.info ("[-] # of operands in op_str != # of capstone operands"
                                                    "(safely ignore this case)")

                        I.normalized_instr = opcode if len(operands_str) == 0 \
                            else str(opcode + '_' + '_'.join(operands))

                        normalized_instrunctions += I.normalized_instr + ', '
                        self.normalized_instns.append(I.normalized_instr)

                    except AttributeError:
                        logging.info ("[+] Ignored: 0x%x@%s" % (I.start, I.parent.parent.name))
                    I = I.next
                B = B.next

            if len(normalized_instrunctions) > 0:
                self.corpus_data[F.name] = normalized_instrunctions[:-2]

            F = F.next
            bar += 1

        bar.finish()
        if failed_disassembly_ctr > 0:
            print("\t[-] (%s) Failed disassembly cases: %d" \
                  % (self.bin_info.bin_name, failed_disassembly_ctr))

    def build_bininfo(self, bin_info):
        """
        Build binary information from raw binary-analysis-tool-generated info
        by creating new instances with:
            unit.Function, unit.BasicBlock, and unit.Instruction
        """

        f_addrs = sorted(self.dumped_funcs.keys())
        F_prev = None

        bar = util.ProgressBar(len(f_addrs), name="BuildInfo: " + bin_info.bin_name)
        for f_idx, f_addr in enumerate(f_addrs):
            # F (unit.IDA_Function) -> UF (unit.Function)
            F = self.dumped_funcs[f_addr]

            UF = self._copy_func_over(F)
            UF.idx = f_idx

            # Allow functions to be traversed
            if F_prev:
                F_prev.next = UF
                UF.prev = F_prev
            else:
                UF.is_first_fn = True

            bb_addrs = sorted(F.bbls.keys())
            B_prev = None
            for bb_addr in bb_addrs:
                # BB (unit.IDA_BasicBlock) -> UB (unit.BasicBlock)
                BB = F.bbls[bb_addr]
                UB = self._copy_bb_over(BB)

                # Allow basic blocks to be traversed within a function
                if B_prev:
                    B_prev.next = UB
                    UB.prev = B_prev

                instn_addrs = sorted(BB.instns.keys())

                I_prev = None
                for instn_addr in instn_addrs:
                    # I (unit.IDA_Instruction) -> UI (unit.Instruction)
                    I = BB.instns[instn_addr]
                    UI = self._copy_instr_over(I)

                    # Allow instructions to be traversed within a basic block
                    if I_prev:
                        I_prev.next = UI
                        UI.prev = I_prev

                    # Update new instruction info
                    UB.instns[instn_addr] = UI
                    I_prev = UI
                    UF.num_ins += 1
                    UB.num_ins += 1

                UF.bbls[bb_addr] = UB
                B_prev = UB

            F_prev = UF
            bin_info.fns[f_addr] = UF
            bar += 1

        bar.finish()
        self.bin_info = bin_info

    def has_bin_info(self):
        if not self.bin_info:
            print ("[-] There is nothing to process...")
            return False
        return True

    def write_bin_info(self, res_path, level=0, resolve_callee=True):
        """
        Dump all binary information according to different levels [0-3]
            Level 0: Function
            Level 1: Function + FunctionInfo (xrefs, immvals, strefs, libcalls)
            Level 2: Function + FunctionInfo + BasicBlock
            Level 3: Function + FunctionInfo + BasicBlock + Instruction
        """

        if self.has_bin_info():
            with open(res_path, "w") as f:
                F = self.bin_info.first_function
                while F:
                    if F.is_linker_func:
                        F = F.next
                        continue

                    recursive = "*" if F.is_recursive else ""
                    imms = F.get_immediates()
                    str_refs = F.get_string_refs()
                    glibc_funcs = F.get_glibc_calls()

                    f.write("[FN_#%d@0x%x:0x%x] %s%s (%dB, %d BBs, %d INs, %d RefTos, %d RefFroms, "
                            "%d ImmVals, %d StrRefs, %d GlibcCalls)\n" \
                            % (F.idx, F.start, F.end, F.name, recursive, F.sz, F.num_bbs, F.num_ins,
                               len(F.call_refs_to) + len(F.jump_refs_to) + len(F.data_refs_to),
                               len(F.refs_from), len(imms), len(str_refs), len(glibc_funcs)))

                    if level > 0:
                        if len(F.call_refs_to) > 0:
                            f.write("\tRefTos (call): %s\n" %
                                    ([self.dumped_funcs[x].name for x in F.call_refs_to]))
                        if len(F.jump_refs_to) > 0:
                            f.write("\tRefTos (jump): %s\n" %
                                    ([self.dumped_funcs[x].name for x in F.jump_refs_to]))
                        if len(F.data_refs_to) > 0:
                            f.write("\tRefTos (data): %s\n" %
                                    ([self.dumped_funcs[x].name for x in F.data_refs_to]))
                        if len(F.refs_from) > 0:
                            f.write("\tRefFroms: %s\n" %
                                    ([self.dumped_funcs[x].name for x in F.refs_from]))

                        if len(imms) > 0:
                            f.write("\tImmVals: %s\n" % (imms))
                        if len(str_refs) > 0:
                            f.write("\tStrRefs: %s\n" % (str_refs))
                        if len(glibc_funcs) > 0:
                            f.write("\tGlibcCalls: %s\n" % (glibc_funcs))

                    if level > 1:
                        bb = F.first_basic_block
                        while bb:
                            f.write("\tBB@0x%x:0x%x (%dB, %d INs)\n" \
                                    % (bb.start, bb.end, bb.sz, len(bb.instns)))

                            if level > 2:
                                ins = bb.first_instruction
                                while ins:
                                    if not ins.is_nop:
                                        normalized_instr = ins.normalized_instr
                                        if resolve_callee and ins.callee > 0:
                                            try:
                                                normalized_instr = "call_" + self.bin_info.fns[ins.callee].name
                                            except KeyError:
                                                pass
                                        f.write("\t\tIN@0x%x (%dB) %s" \
                                                % (ins.start, ins.sz, normalized_instr))
                                        if len(ins.ref_string) > 0:
                                            import glibc
                                            glibc_mark = " (GLIBC)" if ins.ref_string in glibc.function_list else ""
                                            f.write(" '%s'%s" % (ins.ref_string, glibc_mark))
                                        f.write("\n")
                                    ins = ins.next
                            bb = bb.next
                    F = F.next

            print ("\t[+] Dumped info: %s (at the level %d)" % (res_path, level))

    def pickle_dump(self, BS):
        # This function contains quite a few redudant operations
        # due to the failure of pickling instances with pointers

        BS.dir_name = self.bin_info.dir_name
        BS.bin_name = self.bin_info.bin_name
        BS.compiler = self.bin_info.compiler_info_label
        BS.opt_level = self.bin_info.opt_level_label

        BS.num_fns = self.bin_info.num_fns
        BS.num_bbs = self.bin_info.num_bbs
        BS.num_ins = self.bin_info.num_ins
        
        bar = util.ProgressBar(self.bin_info.num_fns,
                               name="Data pickling: " + self.bin_info.bin_name)
        fn = self.bin_info.first_function
        
        # Each summary has a hierarchy to access fn/bb/ins information
        while fn:
            BSFS = unit.FunctionSummary()
            fn_idx = self.Fs.index(fn.name)

            # Basic function information
            BSFS.fn_idx = fn_idx
            BSFS.fn_identifier = "F" + str(fn_idx)
            BSFS.fn_name = fn.name
            BSFS.fn_start = fn.start
            BSFS.fn_end = fn.end
            BSFS.fn_size = fn.sz
            BSFS.fn_num_bbs = fn.num_bbs
            BSFS.fn_num_ins = fn.num_ins
            BSFS.is_linker_func = fn.is_linker_func

            # Function call graph information
            BSFS.ref_tos_by_call = [self.Fs.index(self.dumped_funcs[x].name) for x in fn.call_refs_to] \
                if len(fn.call_refs_to) > 0 else []
            BSFS.ref_tos_by_jump = [self.Fs.index(self.dumped_funcs[x].name) for x in fn.jump_refs_to] \
                if len(fn.jump_refs_to) > 0 else []
            BSFS.ref_tos_by_data = [self.Fs.index(self.dumped_funcs[x].name) for x in fn.data_refs_to] \
                if len(fn.data_refs_to) > 0 else []
            BSFS.ref_froms = [self.Fs.index(self.dumped_funcs[x].name) for x in fn.refs_from] \
                if len(fn.refs_from) > 0 else []
            BSFS.num_ref_tos = len(BSFS.ref_tos_by_call) + len(BSFS.ref_tos_by_jump) \
                               + len(BSFS.ref_tos_by_data)
            BSFS.num_ref_froms = len(BSFS.ref_froms)

            # Function signature information
            BSFS.imms = fn.get_immediates()
            BSFS.str_refs = fn.get_string_refs()
            BSFS.glibc_funcs = fn.get_glibc_calls()
            BSFS.is_recursive = True if fn.is_recursive else False
            BSFS.fn_num_imms = len(BSFS.imms)
            BSFS.num_str_refs = len(BSFS.str_refs)
            BSFS.num_glibc_funcs = len(BSFS.glibc_funcs)

            bb = fn.first_basic_block
            bb_idx = 0

            while bb:
                BSBS = unit.BasicBlockSummary()
                bb_idx += 1

                BSBS.bb_idx = bb_idx
                BSBS.bb_identifier = BSFS.fn_identifier + "_B" + str(bb_idx)

                BSBS.bb_start = bb.start
                BSBS.bb_end = bb.end
                BSBS.bb_size = bb.sz
                BSBS.bb_num_ins = len(bb.instns)

                ins = bb.first_instruction
                ins_idx = 0

                while ins:
                    BSIS = unit.InstructionSummary()
                    ins_idx += 1

                    # Basic instruction information
                    BSIS.ins_idx = ins_idx
                    BSIS.ins_identifier = BSBS.bb_identifier + "_I" + str(ins_idx)
                    BSIS.ins_start = ins.start
                    BSIS.ins_end = ins.end
                    BSIS.ins_size = ins.sz
                    BSIS.ins_opcode = ins.opcode
                    BSIS.ins_operands = ins.operands_str

                    BSIS.is_nop = ins.is_nop
                    BSIS.normalized_instr = ins.normalized_instr
                    BSFS.normalized_instrs.append(ins.normalized_instr)
                    BSFS.raw_bytes += ins.raw_bytes

                    # Instruction signature information
                    if len(ins.imms) > 0:
                        BSIS.ins_imms = ins.imms
                        BSIS.has_imms = True

                    if len(ins.ref_string) > 0:
                        BSIS.ref_string = ins.ref_string
                        BSIS.has_ref_string = True

                    if ins.ref_string in glibc.function_list:
                        BSIS.glibc_func = ins.glibc_func
                        BSIS.has_glibc_call = True

                    BSBS.ins_summaries.append(BSIS)
                    # Go to the next instruction
                    ins = ins.next

                BSFS.bbs_summaries.append(BSBS)
                # Go to the next basic block
                bb = bb.next

            BS.fns_summaries.append(BSFS)
            # Go to the next function
            fn = fn.next
            bar += 1

        bar.finish()

        return BS

    def json_dump(self, res_path):
        """
        Dump all binary information as a json format
        Example of a json dump file
        {"FileName" : {
            "F#1" : {
                    "is_recursive": [false],
                    "glibc_funcs": [],
                    "fn_end": 18185,
                    "str_refs": ["vsftpd: must be started as root (see run_as_launching_user option)"],
                    "num_glibc_funcs": 0,
                    "ref_froms": ["die", "vsf_sysutil_running_as_root"],
                    "fn_start": 18160,
                    "fn_idx": 2,
                    "num_ref_tos": 1,
                    "num_str_refs": 1,
                    "ref_tos_by_data": [],
                    "num_imms": 0,
                    "num_ins": 9,
                    "num_bbs": 3,
                    "fn_size": 25,
                    "ref_tos_by_call": ["main"],
                    "fn_name": "die_unless_privileged",
                    "imms": [],
                    "ref_tos_by_jump": [],
                    "num_ref_froms": 2,
                    "bbs_info": {
                        "B#1": {
                            "bb_idx": 1,
                            "bb_end": 18170,
                            "bb_start": 18160,
                            "num_ins": 4,
                            "ins_info": {
                                "I#1": {
                                    "ins_normalized": "push rax",
                                    "ins_size": 1,
                                    "ins_opcode": "push",
                                    "ins_end": 18161,
                                    "has_imms": false,
                                    "has_ref_string": false,
                                    "ins_operands": "rax",
                                    "has_glibc_call": false,
                                    "ref_string": "",
                                    "ins_start": 18160,
                                    "imms": [],
                                    "ins_idx": 1
                                }, ...
                            }, ...
                    }
            }
        }
        """
        if self.has_bin_info():
            bar = util.ProgressBar(self.bin_info.num_fns - self.bin_info.num_linker_func,
                                   name="JSON dump: " + self.bin_info.bin_name)

            F = self.bin_info.first_function
            json_func_data = {}
            json_func_idx = 0
            while F:
                if F.is_linker_func:
                    F = F.next
                    continue

                json_func_idx += 1
                json_func_identifier = "F" + str(json_func_idx)

                # Collect the function information
                ref_tos_by_call = [self.dumped_funcs[x].name for x in F.call_refs_to] \
                    if len(F.call_refs_to) > 0 else []
                ref_tos_by_jump = [self.dumped_funcs[x].name for x in F.jump_refs_to] \
                    if len(F.jump_refs_to) > 0 else []
                ref_tos_by_data = [self.dumped_funcs[x].name for x in F.data_refs_to] \
                    if len(F.data_refs_to) > 0 else []
                ref_froms = [self.dumped_funcs[x].name for x in F.refs_from] \
                    if len(F.refs_from) > 0 else []

                imms = F.get_immediates()
                str_refs = F.get_string_refs()
                glibc_funcs = F.get_glibc_calls()
                is_recursive = True if F.is_recursive else False

                bb = F.first_basic_block
                json_bb_data = {}
                json_bb_idx = 0
                while bb:
                    json_bb_idx += 1
                    json_bb_identifier = json_func_identifier + "_B" + str(json_bb_idx)
                    ins = bb.first_instruction

                    json_ins_data = {}
                    json_ins_idx = 0

                    while ins:
                        if not ins.is_nop:

                            json_ins_idx += 1

                            has_imms = False
                            has_ref_string = False
                            has_glibc_call = False

                            normalized_instr = ins.normalized_instr

                            if len(ins.imms) > 0:
                                has_imms = True

                            if len(ins.ref_string) > 0:
                                has_ref_string = True

                            if ins.ref_string in glibc.function_list:
                                has_glibc_call = True

                            json_ins_identifier = json_bb_identifier + "_I" + str(json_ins_idx)
                            json_ins_data[json_ins_identifier] = {
                                # Collect the basic instruction information
                                'ins_idx': json_ins_idx,                # instruction index
                                'ins_start': hex(ins.start),            # beginning address of an instruction
                                'ins_end': hex(ins.end),                # ending address of an instruction
                                'ins_size': ins.sz,                     # instruction size in bytes
                                'ins_opcode': ins.opcode,               # opcode of an instruction
                                'ins_operands': ins.operands_str,       # operands of an instruction (>=0)
                                'ins_normalized': normalized_instr,     # normalized instruction

                                # Instruction signature information
                                'has_imms': has_imms,               # mark if an instruction has an immediate
                                'ins_imms': ins.imms,               # list of immediate values if any
                                'has_ref_string': has_ref_string,   # mark if an instruction has a reference string
                                'ref_string': ins.ref_string,       # list of reference strings if any
                                'has_glibc_call': has_glibc_call,   # mark if an instruction calls a glibc function
                            }

                        # Go to the next instruction
                        ins = ins.next

                    json_bb_data[json_bb_identifier] = {
                        # Collect the basic block information
                        'bb_idx': json_bb_idx,              # basic block index
                        'bb_start': hex(bb.start),          # beginning address of a basic block
                        'bb_end': hex(bb.end),              # ending address of a basic block
                        'bb_size': bb.sz,                   # basic block size in bytes
                        'bb_num_ins': len(bb.instns),       # number of instructions that belongs to a basic block
                        'ins_info': json_ins_data,          # instruction information
                    }

                    # Go to the next basic block
                    bb = bb.next

                json_func_data[json_func_identifier] = {
                    # Basic information
                    'fn_idx': json_func_idx,    # function index
                    'fn_name': F.name,                      # function name
                    'fn_start': hex(F.start),               # beginning address of a function
                    'fn_end': hex(F.end),                   # ending address of a function
                    'fn_size': F.sz,                        # function size in bytes
                    'fn_num_bbs': F.num_bbs,                # number of basic blocks that belong to a function
                    'fn_num_ins': F.num_ins,                # number of instructions that belong to a function

                    # Call graph information
                    'num_ref_tos': len(ref_tos_by_call) + len(ref_tos_by_jump) \
                                   + len(ref_tos_by_data),  # number of incoming edges
                    'num_ref_froms': len(ref_froms),        # number of outgoing edges
                    'ref_tos_by_call': ref_tos_by_call,     # list of incoming edges by a call invocation if any
                    'ref_tos_by_jump': ref_tos_by_jump,     # list of incoming edges by a jump instruction if any
                    'ref_tos_by_data': ref_tos_by_data,     # list of incoming edges by a data pointer if any
                    'ref_froms': ref_froms,                 # list of outgoing edges if any

                    # Function signature information
                    'fn_num_imms': len(imms),               # number of immediate values
                    'num_str_refs': len(str_refs),          # number of string references
                    'num_glibc_funcs': len(glibc_funcs),    # number of glibc functions
                    'fn_imms': imms,                        # list of immediate values if any
                    'str_refs': str_refs,                   # list of string references if any
                    'glibc_funcs': glibc_funcs,             # list of glibc functions if any
                    'is_recursive': is_recursive,           # mark if a function is recursive

                    # Basic block and instruction information that belongs to a function
                    'bbs_info': json_bb_data,               # basic block information
                }

                # Go to the next function
                F = F.next
                bar += 1

            bar.finish()

            with open(res_path, 'w') as json_out:
                json.dump(json_func_data, json_out)

            print ("\t[+] Dumped json: %s " % (res_path))

    # Format: (normalized instructions per function, label)
    # Label should be one of
    #   {clang-O0, clang-O1, clang-O2, clang-O3, gcc-O0, gcc-O1, gcc-O2, gcc-O3}
    def generate_learning_data(self):
        """
        TAB-separated (bin_name, fn_name, normalized instructions, label)
        """

        if self.bin_info.compiler_info_label and self.bin_info.opt_level_label:
            label = self.bin_info.compiler_info_label + ' ' + self.bin_info.opt_level_label + '\n'
        else:
            label = "N/A\n"

        corpus_ctr = len(self.corpus_data)
        bar = util.ProgressBar(corpus_ctr, name="Corpus generation: " + self.bin_info.bin_name)

        corpus_data = ''
        for func_name in self.corpus_data:
            corpus_data += '\t'.join([self.bin_info.bin_name, func_name,
                                      self.corpus_data[func_name], label])
            bar += 1

        bar.finish()
        corpus_voca = self.get_normalized_ins_ctr()

        return (corpus_ctr, corpus_data, corpus_voca)

if __name__ == '__main__':
    import pickle

    target = sys.argv[1]
    pkl_dir = sys.argv[2]

    file_name = os.path.basename(target)
    ida_dmp_path = target + ".dmp.gz"
    pkl_dmp_path = os.path.join(pkl_dir,file_name + ".pkl")
    txt_dmp_path = os.path.join(pkl_dir,file_name + ".info.txt")

    opt_level = file_name.split("-")[-1]
    bin_info = unit.Binary_Info(target)
    bin_info.compiler_info_label = file_name.split("-")[-2] 
    bin_info.opt_level_label = opt_level

    print("\t[+] Loading %s..." % ida_dmp_path)
    nn = Normalization(util.load_from_dmp(ida_dmp_path))
    nn.build_bininfo(bin_info)
    nn.disassemble_and_normalize_instructions(normalization_level=3)
    pickle.dump(nn.pickle_dump(unit.BinarySummary()), open(pkl_dmp_path, 'wb'))
    nn.write_bin_info(txt_dmp_path, level=3, resolve_callee=True)
