# This file defines opcode and operands for normalization

X86_CALL = {'call': 0xe8}
X86_JMP = {
    'ja':   0x870f,
    'jae':  0x830f,
    'jb':   0x820f,
    'jbe':  0x860f,
    'jc':   0x820f,
    'je':   0x840f,
    'jg':   0x8f0f,
    'jge':  0x8d0f,
    'jl':   0x8c0f,
    'jle': 0x8e0f,
    'jmp': 0xe9,
    'jna': 0x860f,
    'jnae': 0x820f,
    'jnb': 0x830f,
    'jnbe': 0x870f,
    'jnc': 0x830f,
    'jne': 0x850f,
    'jng': 0x8e0f,
    'jnge': 0x8c0f,
    'jnl': 0x8d0f,
    'jnle': 0x8f0f,
    'jno': 0x810f,
    'jnp': 0x8b0f,
    'jns': 0x890f,
    'jnz': 0x850f,
    'jo': 0x800f,
    'jp': 0x8a0f,
    'jpe': 0x8a0f,
    'jpo': 0x8b0f,
    'js': 0x880f,
    'jz': 0x840f,
}

X86_CALL_BYTES = {0xE8: 'call'}
X86_JMP_BYTES = {
   0xE3: 'jmp',
   0xE9: 'jmp',
   0xEB: 'jmp',
   0x70: 'jo',
   0x71: 'jno',
   0x72: 'jb',
   0x73: 'jae',
   0x74: 'je',  # 'jz'
   0x75: 'jne', # 'jnz'
   0x76: 'jbe',
   0x77: 'jnbe',
   0x78: 'js',
   0x79: 'jns',
   0x7A: 'jp',
   0x7B: 'jnp',
   0x7C: 'jl',
   0x7D: 'jnl',
   0x7E: 'jle',
   0x7F: 'jg',
   0x800F: 'jo',
   0x810F: 'jno',
   0x820F: 'jb',  # 'jc', 'jnae'
   0x830F: 'jnb', # 'jnc', 'jae'
   0x840F: 'jz',  # 'je'
   0x850F: 'jnz', # 'jne'
   0x860F: 'jbe', # 'jna'
   0x870F: 'ja',  # 'jnbe'
   0x880F: 'js',
   0x890F: 'jns',
   0x8A0F: 'jp',  # 'jpe'
   0x8B0F: 'jnp', # 'jno'
   0x8C0F: 'jl',  # 'jnge'
   0x8D0F: 'jge', # 'jnl'
   0x8E0F: 'jle', # 'jng'
   0x8F0F: 'jg',  # 'jnle'
}

# https://www.tortall.net/projects/yasm/manual/html/arch-x86-registers.html
X86_REG_OPERANDS = {
    # 165 registers/pointer types mapping to 25 groups: 2,947 unique instructions
    # 1-byte registers: 10 + 8 = 18
    "al": "reg1", "bl": "reg1", "cl": "reg1", "dl": "reg1",
    "ah": "reg1", "bh": "reg1", "ch": "reg1", "dh": "reg1",
    "sil": "reg1", "dil": "reg1",
    "r8b": "reg1", "r9b": "reg1", "r10b": "reg1", "r11b": "reg1",
    "r12b": "reg1", "r13b": "reg1", "r14b": "reg1", "r15b": "reg1",

    # 2-bytes registers: 6 + 8 = 14
    "ax": "reg2", "bx": "reg2", "cx": "reg2", "dx": "reg2",
    "si": "reg2", "di": "reg2",
    "r8w": "reg2", "r9w": "reg2", "r10w": "reg2", "r11w": "reg2",
    "r12w": "reg2", "r13w": "reg2", "r14w": "reg2", "r15w": "reg2",

    # 4-bytes registers: 6 + 8 = 14
    "eax": "reg4", "ebx": "reg4", "ecx": "reg4", "edx": "reg4",
    "esi": "reg4", "edi": "reg4",
    "r8d": "reg4", "r9d": "reg4", "r10d": "reg4", "r11d": "reg4",
    "r12d": "reg4", "r13d": "reg4", "r14d": "reg4", "r15d": "reg4",

    # 8-bytes registers: 6 + 8 = 14
    "rax": "reg8", "rbx": "reg8", "rcx": "reg8", "rdx": "reg8",
    "rsi": "reg8", "rdi": "reg8",
    "r8": "reg8",  "r9": "reg8", "r10": "reg8", "r11": "reg8",
    "r12": "reg8", "r13": "reg8", "r14": "reg8", "r15": "reg8",

    # https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
    # AVX (AVX, also known as Sandy Bridge New Extensions) registers: 104
    "mm0": "regmm", "mm1": "regmm", "mm2": "regmm", "mm3": "regmm",
    "mm4": "regmm", "mm5": "regmm", "mm6": "regmm", "mm7": "regmm",
    "xmm0": "regxmm", "xmm1": "regxmm", "xmm2": "regxmm", "xmm3": "regxmm",
    "xmm4": "regxmm", "xmm5": "regxmm", "xmm6": "regxmm", "xmm7": "regxmm",
    "xmm8": "regxmm", "xmm9": "regxmm", "xmm10": "regxmm", "xmm11": "regxmm",
    "xmm12": "regxmm", "xmm13": "regxmm", "xmm14": "regxmm", "xmm15": "regxmm",
    "xmm16": "regxmm", "xmm17": "regxmm", "xmm18": "regxmm", "xmm19": "regxmm",
    "xmm20": "regxmm", "xmm21": "regxmm", "xmm22": "regxmm", "xmm23": "regxmm",
    "xmm24": "regxmm", "xmm25": "regxmm", "xmm26": "regxmm", "xmm27": "regxmm",
    "xmm28": "regxmm", "xmm29": "regxmm", "xmm30": "regxmm", "xmm31": "regxmm",
    "ymm0": "regymm", "ymm1": "regymm", "ymm2": "regymm", "ymm3": "regymm",
    "ymm4": "regymm", "ymm5": "regymm", "ymm6": "regymm", "ymm7": "regymm",
    "ymm8": "regymm", "ymm9": "regymm", "ymm10": "regymm", "ymm11": "regymm",
    "ymm12": "regymm", "ymm13": "regymm", "ymm14": "regymm", "ymm15": "regymm",
    "ymm16": "regymm", "ymm17": "regymm", "ymm18": "regymm", "ymm19": "regymm",
    "ymm20": "regymm", "ymm21": "regymm", "ymm22": "regymm", "ymm23": "regymm",
    "ymm24": "regymm", "ymm25": "regymm", "ymm26": "regymm", "ymm27": "regymm",
    "ymm28": "regymm", "ymm29": "regymm", "ymm30": "regymm", "ymm31": "regymm",
	"zmm0": "regzmm", "zmm1": "regzmm", "zmm2": "regzmm", "zmm3": "regzmm",
    "zmm4": "regzmm", "zmm5": "regzmm", "zmm6": "regzmm", "zmm7": "regzmm",
    "zmm8": "regzmm", "zmm9": "regzmm", "zmm10": "regzmm", "zmm11": "regzmm",
    "zmm12": "regzmm", "zmm13": "regzmm", "zmm14": "regzmm", "zmm15": "regzmm",
    "zmm16": "regzmm", "zmm17": "regzmm", "zmm18": "regzmm", "zmm19": "regzmm",
    "zmm20": "regzmm", "zmm21": "regzmm", "zmm22": "regzmm", "zmm23": "regzmm",
    "zmm24": "regzmm", "zmm25": "regzmm", "zmm26": "regzmm", "zmm27": "regzmm",
    "zmm28": "regzmm", "zmm29": "regzmm", "zmm30": "regzmm", "zmm31": "regzmm",

    # Special registers: 10 + 46 = 56
    # [stack/base/instruction | ST | CR/DR / segment] registers
    "spl": "sp1", "bpl": "bp1",
    "sp": "sp2", "bp": "bp2",
    "esp": "sp4", "ebp": "bp4",
    "rsp": "sp8", "rbp": "bp8",
    "eip": "ip4", "rip": "ip8",

    "cr0": "regcr", "cr1": "regcr", "cr2": "regcr", "cr3": "regcr",
    "cr4": "regcr", "cr5": "regcr", "cr6": "regcr", "cr7": "regcr",
    "cr8": "regcr", "cr9": "regcr",  "cr10": "regcr", "cr11": "regcr",
    "cr12": "regcr", "cr13": "regcr", "cr14": "regcr", "cr15": "regcr",

    "dr0": "regdr", "dr1": "regdr", "dr2": "regdr", "dr3": "regdr",
    "dr4": "regdr", "dr5": "regdr", "dr6": "regdr", "dr7": "regdr",
    "dr8": "regdr", "dr9": "regdr", "dr10": "regdr", "dr11": "regdr",
    "dr12": "regdr", "dr13": "regdr", "dr14": "regdr", "dr15": "regdr",

    "st(0)": "regst", "st(1)": "regst", "st(2)": "regst", "st(3)": "regst",
    "st(4)": "regst", "st(5)": "regst", "st(6)": "regst", "st(7)": "regst",

    "cs": "regcs", "ds": "regds",
    "es": "reges", "fs": "regfs",
    "gs": "reggs", "ss": "regss",
}

X86_PTR_OPERANDS = {
    # Pointer types: 9
    # https://github.com/aquynh/capstone/blob/0e6390c85175d31da0bef005d44abca9b4eb7c7a/arch/X86/X86IntelInstPrinter.c
    "byte": "memptr1",  # byte ptr
    "word": "memptr2",  # word ptr
    "dword": "memptr4",  # dword ptr
    "qword": "memptr8",  # qword ptr
    "ptr": "memptr8",  # ptr
    "tbyte": "memptr10",  # tbyte ptr
    "xword": "memptr10",  # xword ptr
    "xmmword": "memptr16",  # xmmword ptr
    "ymmword": "memptr32",  # ymmword ptr
    "zmmword": "memptr64"  # zmmword ptr
}
