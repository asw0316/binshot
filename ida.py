import idautils
import idaapi
import idc
import ida_bytes
import ida_pro

import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unit import IDA_Function
from unit import IDA_BasicBlock
from unit import IDA_Instruction
import glibc

import pickle
from bz2 import BZ2File
code_heads = set()

class IDA(object):
    """
    This class has been designed for reconstructing the analysis from IDA
    Useful IDA-provided APIs
        IDC, IDA_BYTES
            idc.Byte(ea): Returns the byte at the given address
            idc.get_bytes(ea, num): Returns num of bytes at the given address
            idc.ItemSize(ea): Returns the size of the given address
            idc.SegStart(ea): Returns the beginning of the segment that ea belongs to
            idc.SegEnd(ea): Returns the end of the segment that ea belongs to
            ida_bytes.get_item_end(ea): Returns the end of the instruction (or item)
            idc.get_func_name(ea): Returns the name of the function that ea belongs to
            idc.GetDisasm(ea): Returns the disassembly at ea
            idc.GetType(fn_ea): Returns the type of the function at fn_ea
            idc.get_segm_name(ea): Returns the name of the section that ea belongs to
        IDAUtils
            idautils.Functions(): Returns the addresses of all functions
            idautils.Entries(): Returns a list of entry points or exports (idx, start, end, fname)
            idautils.XrefsFrom(ea, flags=[idaapi.ALL|idaapi.XREF_FAR|idaapi.XREF_DATA])
                : Returns all references from ea as a list of idautils._xref objects
            idautils.XrefsTo(ea, flags=[idaapi.ALL|idaapi.XREF_FAR|idaapi.XREF_DATA])
                : Returns all references to ea as a list of idautils._xref objects
            idautils.Heads(start_ea, end_ea): Returns a list of heads btn start/end
        IDAAPI
            idaapi.BADADDR: Returns 0xffffffff or 0xffffffffffffffffff
            idaapi.get_func(ea): Returns a idaapi.func_t
            idaapi.get_input_file_path(): Returns the abspath of the input file
            idaapi.get_fileregion_offset(ea): Returns the file offset from ea
            idaapi.FlowChart(idaapi.get_func(ea)): Returns all idaapi.BasicBlock objects
            idaapi.is_call_insn(ea): Returns true if ea has a call invocation
            idaapi.is_indirect_jump_insn(ea): Returns true if ea is an indirection jump
            idaapi.getseg(ea): Returns a idaapi.segment_t
            idaapi.decompile(ea): Returns decompiled code at ea
    """
    def __init__(self):
        self.all_funs = dict()
        self.all_bbls = dict()
        self.all_insns = dict()

    def update_refs(self, ea):
        """
        Update the following references (edges) for call graph construction
            1) the set of functions to reach here (as a callee)
            2) the set of functions to call from here (as a caller)
        Three types expected (idautils.XrefTypeName(xref.type))
            1) call instruction ('Code_Near_Call')
            2) jump instruction ('Code_Near_Jump')
            3) data region; an entry from a jump table ('Data_Offset')
        :param ea:
        :return:
        """

        # Make sure 'ea' must be always the beginning of a function
        for xref in idautils.XrefsTo(ea, flags=idaapi.XREF_ALL):
            xref_addr = xref.frm
            # Consider the functions defined within a binary
            if idc.get_segm_name(xref_addr) == ".text":
                # caller update (refs_from)
                try:
                    caller_start = idaapi.get_func(xref_addr).start_ea
                    caller = self.all_funs[caller_start]
                    caller.refs_from.add(ea)

                    # callee update (refs_to)
                    callee = self.all_funs[ea]
                    if caller == callee:
                        callee.is_recursive = True

                    if xref.type == 17:  # 'Code_Near_Call'
                        callee.call_refs_to.add(caller_start)
                    elif xref.type == 19:  # 'Code_Near_Jump'
                        callee.jump_refs_to.add(caller_start)
                    elif xref.type == 1:  # 'Data_Offset'
                        callee.data_refs_to.add(caller_start)
                except:
                    # Possibly IDA Pro did not correctly detect the function end of xref_addr
                    # Most of the time, this does not raise an issue in call graph because
                    #   it often happens when a function has just an early return.
                    print ("[-] Missing a function start @%s" % hex(xref_addr))
                    pass

    def get_string(self, ref_addr, limit=30):
        """
        Return a string at the refrenced addr
        """
        str = ""

        byte_ctr = 0
        while True:
            if idc.Byte(ref_addr) == 0 or byte_ctr > limit:
                break
            str += chr(idc.Byte(ref_addr))
            ref_addr += 1
            byte_ctr += 1

        return str

    def run(self):
        """
        Collect all info from analysis in IDA Pro
        :return:
        """
        # Expect f_ea has been sorted
        f_idx = 0
        for f_ea in idautils.Functions():
            if idc.get_segm_name(f_ea) == ".text":
                ida_ft = idaapi.get_func(f_ea)
                f_start, f_end = ida_ft.start_ea, ida_ft.end_ea

                # Corner case (maybe an IDA Pro bug?)
                if f_start == f_end:
                    continue

                F = IDA_Function(f_start, f_end)
                F.idx = f_idx
                F.name = idc.get_func_name(f_ea)
                F.no_return = idc.get_func_attr(f_ea, FUNCATTR_FLAGS) & idc.FUNC_NORET

                # Check if any alignment exists at the function level
                if 'align' in idc.GetDisasm(f_end):
                    align_bytes = abs(ida_bytes.get_item_end(f_end) - f_end)
                    F.align_sz = align_bytes

                bbs = idaapi.FlowChart(ida_ft)
                for bb in bbs:
                    bb_start, bb_end = bb.start_ea, bb.end_ea

                    # Corner case (maybe an IDA Pro bug?)
                    if bb_start == bb_end:
                        continue

                    BB = IDA_BasicBlock(bb_start, bb_end)

                    # Check if any alignment exists between basic blocks
                    if 'align' in idc.GetDisasm(bb_end):
                        align_bytes = abs(ida_bytes.get_item_end(bb_end) - bb_end)
                        BB.align_sz = align_bytes

                    if bb_start == F.start:
                        BB.is_entry = True

                    if bb_end == F.end:
                        BB.is_exit = True

                    BB.parent = F

                    insn_addr = bb_start
                    while insn_addr < bb_end:
                        insn_start, insn_end = insn_addr, ida_bytes.get_item_end(insn_addr)
                        I = IDA_Instruction(insn_start, insn_end)
                        I.is_call = idaapi.is_call_insn(insn_start)
                        if I.is_call:
                            try:
                                xref = [x for x in idautils.XrefsFrom(insn_start) if x.type == 17][0]
                                I.callee = xref.to
                            except:
                                print("[-] Failed to find a callee @%s (Likely a dynamic call)" \
                                      % hex(insn_start))
                                pass
                        I.raw_bytes = idc.get_bytes(insn_start, I.sz)
                        I.parent = BB

                        BB.instns[insn_start] = I
                        insn_addr = insn_end
                        self.all_insns[I.start] = I

                    F.bbls[bb_start] = BB

                self.all_funs[f_start] = F
                f_idx += 1

        # Once construct all functions, update cross references (call graph)
        for f_ea in idautils.Functions():
            if idc.get_segm_name(f_ea) == ".text":
                self.update_refs(f_ea)

        for str_ea in idautils.Strings():
            str_addr = str_ea.ea
            ref_str = str(str_ea).strip()

            if idc.get_segm_name(str_addr) == ".rodata":
                for xref in idautils.XrefsTo(str_addr, flags=idaapi.XREF_DATA):
                    # Note that here ignores any reference other than a code section
                    xref_addr = xref.frm
                    if idc.get_segm_name(xref_addr) == '.text':
                        try:
                            I = self.all_insns[xref_addr]
                            #print ("0x%x;%s (%dB) from Ins[0x%x:0x%x]" %
                            #       (str_addr, ref_str, str_item.length, I.start, I.end))
                            I.ref_string = ref_str
                        # Sometimes IDA has a wrong function boundary, missing an instruction
                        except KeyError:
                            print ("[-] Missing an instruction lookup @0x%x" % (xref_addr))

        # Collect all imports (PLT/GOT) from glibc (library calls)
        for ea, name in idautils.Names():
            # IDA Pro seems to have two different ways to contain external function pointers
            #   (1) '.got' section ending with a '_ptr' postfix
            #   (2) '.plt' section starting with a '.' prefix
            if idc.get_segm_name(ea) == ".got" and name.endswith('_ptr'):
                glibc_func_name = name[:-4]
                print ("[%s] %s" % (idc.get_segm_name(ea), glibc_func_name))
                if glibc_func_name in glibc.function_list:
                    # Each GOT entry corresponds to a single location in a .plt.got section
                    pltgot_addr = [pltgot.frm for pltgot in idautils.XrefsTo(ea, flags=idaapi.XREF_FAR)][0]

                    # Now track all functions that refer to this entry in the .plt.got
                    # Note that a library address would be resolved at the first call invocation only
                    #   at runtime, however it is safe to ignore it during static analysis
                    for pltgot_xref in idautils.XrefsTo(pltgot_addr, flags=idaapi.XREF_FAR):
                        xref_addr = pltgot_xref.frm
                        if idc.get_segm_name(xref_addr) == '.text':
                            try:
                                I = self.all_insns[xref_addr]
                                # print ("[0x%x] %s -> 0x%x -> 0x%x (%s)" \
                                #       % (ea, name, pltgot_addr, xref_addr, glibc_func_name))
                                I.glibc_func = glibc_func_name
                            # Sometimes IDA has a wrong function boundary, missing an instruction
                            except KeyError:
                                print ("[-] Missing an instruction lookup @0x%x" % (xref_addr))

            if idc.get_segm_name(ea) == ".plt" and name.startswith('.'):
                glibc_func_name = name[1:]
                print ("[%s] %s" % (idc.get_segm_name(ea), glibc_func_name))
                if glibc_func_name in glibc.function_list:
                    plt_xref_addrs = [plt.frm for plt in idautils.XrefsTo(ea, flags=idaapi.XREF_FAR)]
                    for xref_addr in plt_xref_addrs:
                        if idc.get_segm_name(xref_addr) == '.text':
                            try:
                                I = self.all_insns[xref_addr]
                                I.glibc_func = glibc_func_name
                            except KeyError:
                                print("[-] Missing an instruction lookup @0x%x" % (xref_addr))

    def get_funcs(self):
        return self.all_funs

    def func_iter(self, details = False):
        """
        Function iterator for dumping out
        :param details:
        :return:
        """
        for f_addr in sorted(self.all_funs.keys()):
            F = self.all_funs[f_addr]
            BBs = F.bbls
            F.num_bbs = len(BBs)
            if details:
                print (F)
                if len(F.refs_from) > 0:
                    print (" - Refs_from: %s" % [hex(x) for x in F.refs_from])
                if len(F.refs_to) > 0:
                    print (" - Refs_to: %s" % [hex(x) for x in F.refs_to])

            for bb_addr in sorted(BBs.keys()):
                BB = BBs[bb_addr]
                instns = BB.instns
                BB.num_instns = len(instns)

                if details:
                    print ("  %s" % BB)
                    for ins_addr in sorted(instns.keys()):
                        print ("     %s" % instns[ins_addr])
                    if BB.align_sz > 0 and not BB.is_exit:
                        print ("---- (BB align %dB) ----" % BB.align_sz)

            if F.align_sz > 0 and details:
                print ("------ (FN align %dB) ------" % F.align_sz)

            yield F

    def dump_data(self):
        dmp_out = BZ2File(idaapi.get_input_file_path() + '.dmp.gz', 'wb')

        for func in self.func_iter():
            pickle.dump(func, dmp_out)

        pickle.dump(None, dmp_out)
        dmp_out.close()

    def dump_noret(self):
        out = idaapi.get_input_file_path() + '_noret.txt'
        with open(out, 'w') as f:
            for fs in sorted(self.all_funs.keys()):
                F = self.all_funs[fs]
                if F.no_return:
                    f.write("0x%08x:0x%08x, %s\n" % (F.start, F.end, F.name))

    def dump_function_boundary(self):
        out = idaapi.get_input_file_path() + '_fb.txt'
        with open(out, 'w') as f:
            for fs in sorted(self.all_funs.keys()):
                F = self.all_funs[fs]
                f.write("0x%08x:0x%08x, %s\n" % (F.start, F.end, F.name))

if __name__ == "__main__":
    sys.setrecursionlimit(40000)
    idc.auto_wait()  # Wait for IDA to complete its auto analysis

    print ('-------- DUMP start -----------')
    ida = IDA()
    ida.run()
    ida.dump_data()
    #ida.dump_noret()
    #ida.dump_function_boundary()
    print ('-------- DUMP end -----------')
    ida_pro.qexit(0)  # Exit IDA when done;
