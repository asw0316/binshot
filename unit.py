import os
import logging
import elf

class Section(object):
    def __init__(self):
        """
        This class represents each section in ELF
        """
        self.idx = -1
        self.parser = None      # Pointer to elf.get_section(idx)

        # Section description
        self.type = None        # 'sh_type': "Section Type"
        self.align = 0x0        # 'sh_addralign': "Section Address Align"
        self.offset = 0x0       # 'sh_offset': "Section Offset"
        self.entsize = 0x0      # 'sh_entsize': "Section Entry Size"
        self.name = ''          # 'sh_name': "Section Name"
        self.flags = 0x0        # 'sh_flags': "Section Flags"
        self.sz = 0x0           # 'sh_size': "Section Size"
        self.va = 0x0           # 'sh_addr': "Section VA"
        self.link = 0x0         # 'sh_link': "Section Link"
        self.info = 0x0         # 'sh_info': "Section Info"

        self.start = 0x0
        self.end = 0x0
        self.file_offset = 0x0  # includes the alignment to rewrite a binary
        self.next = None

    def __repr__(self):
        return '[Sec#%2d] FileOff[0x%04x:0x%04x] VA=0x%08x (%s)' \
               % (self.idx, self.start, self.file_offset, self.va, self.name)


class Symbol(object):
    def __init__(self):
        """
        This class represents each symbol in ELF (i.e., .symtab or .dynsym)
        """
        self.idx = -1
        self.parser = None

        # Symbol description
        self.name = ''
        self.type = 0x0
        self.bind = 0x0
        self.other = 0x0
        self.shndx = 0x0
        self.val = 0x0
        self.sz = 0x0

        # Property
        self.is_import = False
        self.is_export = False
        self.file_path = None

    def __repr__(self):
        return '[Sym#%2d] %s (Val=0x%08x, Size=%d)' \
            % (self.idx, self.name, self.val, self.sz)

# Take an instance of a Binary_Info class to hold summary info
class BinarySummary():
    def __init__(self):
        self.dir_name = None
        self.bin_name = None
        self.compiler = None
        self.opt_level = None
        
        self.num_fns = None
        self.num_bbs = None
        self.num_ins = None
        self.fns_summaries = [] # list of summary information for functions

class FunctionSummary():
    def __init__(self):
        # Basic information
        self.fn_idx = -1,  # function index
        self.fn_identifier = None  # function identifier
        self.fn_name = '',  # function name
        self.fn_start = 0x0,  # beginning address of a function
        self.fn_end = 0x0,  # ending address of a function
        self.fn_size = 0  # function size in bytes
        self.fn_num_bbs = 0  # number of basic blocks that belong to a function
        self.fn_num_ins = 0  # number of instructions that belong to a function
        self.is_linker_func = False  # mark if a function is included by a linker
        self.raw_bytes = b''

        # Call graph information
        self.num_ref_tos = 0  # number of incoming edges
        self.num_ref_froms = 0  # number of outgoing edges
        self.ref_tos_by_call = []  # list of incoming edges by a call invocation if any
        self.ref_tos_by_jump = []  # list of incoming edges by a jump instruction if any
        self.ref_tos_by_data = []  # list of incoming edges by a data pointer if any
        self.ref_froms = []  # list of outgoing edges if any

        # Function signature information
        self.fn_num_imms = 0  # number of immediate values
        self.num_str_refs = 0  # number of string references
        self.num_glibc_funcs = 0  # number of glibc functions
        self.imms = []  # list of immediate values if any
        self.str_refs = []  # list of string references if any
        self.glibc_funcs = []  # list of glibc functions if any
        self.is_recursive = False  # mark if a function is recursive

        # Basic block and instruction information that belongs to a function
        self.bbs_summaries = []  # list of summary information for basic blocks

        # Normalized instruction (directly access)
        self.normalized_instrs = []

class BasicBlockSummary():
    def __init__(self):
        # Basic information
        self.bb_idx = -1,
        self.bb_identifier = None
        self.bb_start = 0x0
        self.bb_end = 0x0
        self.bb_size = 0
        self.bb_num_ins = 0
        self.ins_summaries = [] # list of summary information for instructions

class InstructionSummary():
    def __init__(self):
        # Basic information
        self.ins_idx = -1,
        self.ins_identifier = None
        self.ins_start = 0x0
        self.ins_end = 0x0
        self.ins_size = 0

        # Instruction properties
        self.is_nop = False
        self.ins_opcode = None
        self.ins_operands = None
        self.ins_normalized = None

        self.ins_imms = []
        self.has_imms = False
        self.ref_string = ''
        self.has_ref_string = False
        self.glibc_func = None
        self.has_glibc_call = False
        self.normalized_instr = None

class Binary_Info(object):
    """
    Binary class
    """
    def __init__(self, bin_path):
        self.bin_path = bin_path
        self.bin_name = os.path.basename(bin_path)
        self.dir_name = os.path.dirname(bin_path)
        self.fns = dict()
        self.ep = elf.ELFParser(bin_path)

        self.compiler_info_label = None   # Ground truth of a compiler
        self.opt_level_label = None       # Ground truth of an opt level
        self.compiler_info_pred = None    # Prediction of a compiler
        self.opt_level_pred = None        # Prediction of an opt level
        self.num_linker_func = 0          # Number of functions from a linker

    def show_headers(self):
        self.ep.read_elf_headers()

    def show_sections(self):
        self.ep.show_sections()

    def show_relocations(self):
        self.ep.read_relocations()

    def show_imports(self):
        self.ep.read_imports(show=True)

    def show_exports(self):
        self.ep.read_exports(show=True)

    def show_symbols(self):
        self.ep.read_symbols()

    @property
    def first_function(self):
        return self.fns[sorted(self.fns)[0]]

    @property
    def last_function(self):
        return self.fns[sorted(self.fns)[-1]]

    @property
    def num_fns(self):
        return len(self.fns)

    @property
    def num_bbs(self):
        bb_ctr = 0
        ff = self.first_function
        while ff:
            bb_ctr += ff.num_bbs
            ff = ff.next
        return bb_ctr

    @property
    def num_ins(self):
        ins_ctr = 0
        ff = self.first_function
        while ff:
            ins_ctr += ff.num_ins
            ff = ff.next
        return ins_ctr

    def get_function_by_addr(self, f_addr):
        """
        Return a function instance at f_addr
        :param f_addr:
        :return:
        """
        return self.fns[f_addr]

    def __repr__(self):
        return "[%s] (%d FNs, %d BBs, %d INs)" \
               % (self.bin_name, self.num_fns, self.num_bbs, self.num_ins)


class IDA_Function(object):
    """
    IDA_Function Class
    """
    def __init__(self, start, end):
        self.idx = -1
        self.name = ''
        self.demangled = None
        self.start = start
        self.end = end
        self.sz = end - start

        #self.no_return = False

        self.bbls = dict()
        self.num_bbs = len(self.bbls)
        self.ref_strings = list()

        self.is_linker_func = False
        self.is_recursive = False
        self.call_refs_to = set()  # Functions that reach to this function with a call
        self.jump_refs_to = set()  # Functions that reach to this function with a jump
        self.data_refs_to = set()  # Functions that reach to this function with a jump table
        self.refs_from = set()     # Functions that refer from this function
        self.align_sz = 0x0

class Function(IDA_Function):
    """
    Function class extends IDA_Function class
    """
    def __init__(self, start, end):
        IDA_Function.__init__(self, start, end)

        # Debugging info if needed
        self.cu = None      # Compilation unit (module)
        self.line = 0x0     # Line number

        # Properties
        self.num_ins = 0
        self.summary_embedding = None

        # Linked List within a binary
        self.prev = None
        self.next = None

    @property
    def first_basic_block(self):
        return self.bbls[sorted(self.bbls)[0]]

    @property
    def last_basic_block(self):
        return self.bbls[sorted(self.bbls)[-1]]

    def get_immediates(self):
        immediates = []
        bbk = self.first_basic_block
        while bbk:
            ins = bbk.first_instruction
            while ins:
                if len(ins.imms) > 0:
                    immediates.append(ins.imms[0])
                ins = ins.next
            bbk = bbk.next
        return immediates

    def get_string_refs(self):
        string_refs = []
        bbk = self.first_basic_block
        while bbk:
            ins = bbk.first_instruction
            while ins:
                if len(ins.ref_string) > 0:
                    string_refs.append(ins.ref_string)
                ins = ins.next
            bbk = bbk.next
        return string_refs

    def get_glibc_calls(self):
        glibc_funcs = []
        bbk = self.first_basic_block
        while bbk:
            ins = bbk.first_instruction
            while ins:
                if ins.glibc_func:
                    glibc_funcs.append(ins.glibc_func)
                ins = ins.next
            bbk = bbk.next
        return glibc_funcs

    def get_xrefs(self):
        return self.call_refs_to, self.jump_refs_to, self.data_refs_to

    def __repr__(self):
        return "[FN_#%d@0x%x:0x%x] %s (%dB, %d BBs, %d INs)" \
               % (self.idx, self.start, self.end,
                  self.name, self.sz, self.num_bbs, self.num_ins)


class IDA_BasicBlock(object):
    """
    IDA_BasicBlock class
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.sz = end - start

        self.parent = None
        self.instns = dict()
        self.num_ins = len(self.instns)

        # self.instrs = [i for a, i in code.iteritems() if a >= addr and a <= end]
        # self.instrs.sort(key=lambda x: x.start)

        self.is_entry = False
        self.is_exit = False
        self.align_sz = 0x0

class BasicBlock(IDA_BasicBlock):
    """
    BasicBlock class extends IDA_BasicBlock class
    """
    def __init__(self, start, end):
        IDA_BasicBlock.__init__(self, start, end)

        # Properties
        self.summary_embedding = None

        # Linked List within a function
        self.prev = None
        self.next = None

    @property
    def first_instruction(self):
        return self.instns[sorted(self.instns)[0]]

    @property
    def last_instruction(self):
        return self.instns[sorted(self.instns)[-1]]

    def __repr__(self):
        is_entry = "E" if self.is_entry else ""
        is_exit = "X" if self.is_exit else ""
        return "BB@0x%x:0x%x %s%s (%dB, %d INs)" \
               % (self.start, self.end, is_entry, is_exit, self.sz, self.num_ins)


class IDA_Instruction(object):
    """
    IDA_Instruction class
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.sz = end - start

        self.is_call = False
        self.callee = 0x0
        self.parent = None
        self.raw_bytes = b''
        self.ref_string = ''
        self.glibc_func = None

class Instruction(IDA_Instruction):
    """
    Instruction class extends IDA_Instruction class
    """
    def __init__(self, start, end):
        IDA_Instruction.__init__(self, start, end)

        # Instruction information
        self.cs_instr = None
        self.opcode = None
        self.operands_str = None
        self.operands = []
        self.normalized_instr = None
        self.is_nop = False

        # Properties
        self.has_immediate = False
        self.imms = []
        self.embedding = None   # (n) dimensional array

        # Linked List within a basic block
        self.prev = None
        self.next = None

    @property
    def num_imms(self):
        return len(self.imms)

    @property
    def num_operands(self):
        return len(self.operands)

    def __repr__(self):
        is_call = "C" if self.is_call else ""
        return "IN@0x%x:0x%x %s (%dB)" \
               % (self.start, self.end, is_call, self.sz)
