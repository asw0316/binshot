import os, sys
import logging
import unit
import conf as C

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.relocation import RelocationSection
    from elftools.elf.sections import SymbolTableSection
    from elftools.elf.dynamic import DynamicSection
except:
    logging.critical("You need to install the following packages: pyelftools")
    sys.exit(1)

class ELFParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(self.file_path, 'rb')
        self.bin = self.f.read()
        self.elf = ELFFile(self.f)

        self.struct_elf = {
           'e_type': "Object File Type",
           'e_machine': "Architecture",
           'e_entry': "Entry Point VA",
           'e_phoff': "Program header table file offset",
           'e_shoff': "Section header table file offset",
           'e_ehsize': "ELF header size in bytes",
           'e_phentsize': "Program header table entry size",
           'e_phnum': "Program header table entry count",
           'e_shentsize': "Section header table entry size",
           'e_shnum': "Section header table entry count",
           'e_shstrndx': "Section header string table index"
        }

        self.struct_section = {
            'sh_type': "Section Type",
            'sh_addralign': "Section Address Align",
            'sh_offset': "Section Offset",
            'sh_entsize': "Section Entry Size",
            'sh_name': "Section Name",
            'sh_flags': "Section Flags",
            'sh_size': "Section Size",
            'sh_addr': "Section VA",
            'sh_link': "Section Link",
            'sh_info': "Section Info"
        }

        # Relocation Types: Value, Name, Field and Calculation from linux64-ABI
        self.struct_relocation = {
            0: "R_X86_64_NONE",             # none, none
            1: "R_X86_64_64",               # word64, S + A
            2: "R_X86_64_PC32",             # word32, S + A - P
            3: "R_X86_64_GOT32",            # word32, G + A
            4: "R_X86_64_PLT32",            # word32, L + A - P
            5: "R_X86_64_COPY",             # none, none
            6: "R_X86_64_GLOB_DAT",         # wordclass, S
            7: "R_X86_64_JUMP_SLOT",        # wordclass, S
            8: "R_X86_64_RELATIVE",         # wordclass, B + A
            9: "R_X86_64_GOTPCREL",         # word32, G + GOT + A - P
            10: "R_X86_64_32",              # word32, S + A
            11: "R_X86_64_32S",             # word32, S + A
            12: "R_X86_64_16",              # word16, S + A
            13: "R_X86_64_PC16",            # word16, S + A - P
            14: "R_X86_64_8",               # word8, S + A
            15: "R_X86_64_PC8",             # word8, S + A - P
            16: "R_X86_64_DTPMOD64",        # word64
            17: "R_X86_64_DTPOFF64",        # word64
            18: "R_X86_64_TPOFF64",         # word64
            19: "R_X86_64_TLSGD",           # word32
            20: "R_X86_64_TLSLD",           # word32
            21: "R_X86_64_DTPOFF32",        # word32
            22: "R_X86_64_GOTTPOFF",        # word32
            23: "R_X86_64_TPOFF32",         # word32
            24: "R_X86_64_PC64",            # word64, S + A - P (only for LP64)
            25: "R_X86_64_GOTOFF64",        # word64, S + A - GOT (only for LP64)
            26: "R_X86_64_GOTPC32",         # word32, GOT + A - P
            32: "R_X86_64_SIZE32",          # word32, Z + A
            33: "R_X86_64_SIZE64",          # word64, Z + A (only for LP64)
            34: "R_X86_64_GOTPC32_TLSDESC", # word32
            35: "R_X86_64_TLSDESC_CALL",    # none
            36: "R_X86_64_TLSDESC",         # word64 * 2
            37: "R_X86_64_IRELATIVE",       # wordclass, indirect (B + A)
            38: "R_X86_64_RELATIVE64"       # word64, B + A (only for ILP32 executable or shared objects)
        }

        # Sections
        self.sections = list()
        self.text_va = 0x0
        self.section_ranges = dict()

        # Symbols (.symtab)
        self.symbols = list()

        # Imports/Exports (.dynsym)
        self.importedSO = dict()        # ELFParser pointers to directly imported by this ELF
        self.imports = dict()           # (import func name, VA defined in this ELF)
        self.exports = dict()           # (export func name, (shared obj that defines the export, VA))

        self.exports_lookup_by_fn = dict()  # For read_symbols() only
        self.exports_lookup_by_va = dict()  # For read_symbols() only

        self.__extract_section_addrs()

    def __extract_section_addrs(self):
        for s in range(1, self.elf.num_sections()):
            sec = self.elf.get_section(s)
            va = sec['sh_addr']
            if va > 0:
                self.section_ranges[sec.name] = ((va, va + sec['sh_size']))

    def read_elf_headers(self):
        print ('ELF Header (%s)' % self.file_path)
        elf_info = self.elf._parse_elf_header()

        for i in sorted(self.struct_elf.keys()):
            elf_decr = self.struct_elf[i].ljust(35)
            if isinstance(elf_info[i], int):
                val = '(' + hex(elf_info[i]) + ')'
                print ("  %s: %s%s" % (elf_decr, elf_info[i], val.rjust(15)))
            else:
                print ("  %s: %s" % (elf_decr, elf_info[i]))

    def read_relocations(self):
        # There are several different sections for relocation:
        # '.rela.plt', '.rela.dyn', '.rel.plt', '.rel.dyn'
        # The postfix .dyn represents the table for dynamic linker

        for reloc_name in C.RELOC_SECTIONS:
            rel = self.elf.get_section_by_name(reloc_name)
            if isinstance(rel, RelocationSection):
                print ('Relocation Section: %s (%d)' % (reloc_name, rel.num_relocations()))
                # Lookup all entry attributes
                for i, r in enumerate(rel.iter_relocations()):
                    print ('\t[%3d] Offset + Addend: %s +' % (i+1, hex(r['r_offset'])),)
                    if 'rela' in reloc_name:
                        print (r['r_addend'],)
                    print ('\tInfo (Type, Symbol): %s (%s, %s)' \
                          % (hex(r['r_info']), self.struct_relocation[r['r_info_type']],r['r_info_sym']))

    def read_symbols(self, show=False):
        """
        Read all symbols in the symbol table (.symtab)
        :return:
        """
        sym_no = 0
        logging.debug("[+] Start to read all symbols in .symtab@%s..." % self.file_path)
        for sec in self.elf.iter_sections():
            if isinstance(sec, SymbolTableSection) and C.SYMBOL_TABLE_SECTION in sec.name:
                for symbol in sec.iter_symbols():
                    # Create a symbol instance
                    sym = unit.Symbol()
                    sym.parser = symbol
                    sym.name = symbol.name
                    sym.type = symbol['st_info']['type']
                    sym.bind = symbol['st_info']['bind']
                    sym.other = symbol['st_other']['visibility']
                    sym.shndx = symbol['st_shndx']
                    sym.val = symbol['st_value']
                    sym.sz = symbol['st_size']
                    sym.file_path = os.path.basename(self.file_path)

                    sym_no += 1
                    sym.idx = sym_no
                    self.symbols.append(sym)

                    # Import/Export should be mutually exclusive...
                    if self._is_imported(symbol):
                        sym.is_import = True
                    if self._is_exported(symbol):
                        sym.is_export = True
                        self.exports_lookup_by_fn[sym.name] = sym
                        self.exports_lookup_by_va[sym.val] = sym

                    if show:
                        print ("      [%2d] Symbol: %s (Ty=%-7s, Bind=%-6s, Sym_Other=%-7s, Shndx=%4s, Val=0x%x, Sz=0x%x)" % \
                              (sym.idx, sym.name, sym.type, sym.bind, sym.other, sym.shndx, sym.val, sym.sz))
                        #if sym.type == 'STT_FUNC' and sym.val > 0:
                        #    print  ("%s, %s" % (self.file_path, sym.name))

        if sym_no == 0:
            logging.warning("[-] No .symtab has been found...!")
        else:
            logging.debug("[+] Processed %d symbols..." % (sym_no))

    def get_text_section_va(self):
        return self.text_va

    def read_sections(self):
        """
        Read all sections to check the layout of the given binary
        :return:
        """

        elf_info = self.elf._parse_elf_header()

        for i in range(1, self.elf.num_sections()):
            sec = self.elf.get_section(i)
            s = unit.Section()
            s.idx = i
            s.parser = sec
            s.name = sec.name

            if "text"in sec.name:
                self.text_va = sec['sh_addr']

            # The following sections only exist in the memory space
            if s.name == C.TM_CLONE_TABLE_SECTION or s.name == C.BSS_SECTION:
                continue

            s.type = sec['sh_type']
            s.align = sec['sh_addralign']
            s.entsize = sec['sh_entsize']
            s.flags = sec['sh_flags']
            s.sz = sec['sh_size']
            s.va = sec['sh_addr']
            s.start = s.offset = sec['sh_offset']
            s.end = s.start + s.sz
            s.link = sec['sh_link']
            s.info = sec['sh_info']

            self.sections.append(s)

        # Section boundary updates for binary instrumentation purpose
        self.sections = sorted(self.sections, key=lambda s: s.start)
        self.sections[-1].file_offset_end = elf_info['e_shoff']
        for i in range(len(self.sections)-1):
            self.sections[i].next = self.sections[i+1]
            if self.sections[i].end != self.sections[i + 1].start:
                self.sections[i].file_offset = self.sections[i + 1].start
            else:
                self.sections[i].file_offset = self.sections[i].end

        # Workaround; when a section start is not in order from the original file
        tmp_sec = self.sections[0]
        while tmp_sec:
            if tmp_sec.file_offset == 0:
                tmp_sec.file_offset = tmp_sec.start + tmp_sec.sz + 1
                break
            tmp_sec = tmp_sec.next

        return self.sections

    def show_sections(self):
        """
        This function is to check if this script works independently only
        :return:
        """

        self.read_sections()
        print ('Found %s sections: ' % len(self.sections))
        for idx, s in enumerate(self.sections):
            print ('  [%2d] Section %s' % (idx, s.name))
            for ss in sorted(self.struct_section.keys()):
                sec_desc = self.struct_section[ss].ljust(25)
                print ('\t%s : %s' % (sec_desc, s.parser[ss]))

    def __get_symbol_info(self, symbol):
        return symbol['st_value'], symbol['st_shndx'],\
                symbol['st_info']['bind'], symbol['st_info']['type']

    def _is_imported(self, symbol):
        """
        Check if a symbol has been imported
        :param symbol:
        :return:
        """
        def __symvalCheck(val):
            return val == 0x0 or self.is_within_section('.plt', val)

        symval, symndx, symbind, symtype = self.__get_symbol_info(symbol)

        # symbol bind in {STB_GLOBAL, STB_WEAK} type in {STT_OBJ, STT_FUNC, STT_LOOS}
        return __symvalCheck(symval) and (isinstance(symndx, str) and 'UND' in symndx) and \
                        ('GLOBAL' in symbind or 'WEAK' in symbind) and \
                        ('OBJ' in symtype or 'FUNC' in symtype or 'LOOS' in symtype)

    def _is_exported(self, symbol):
        """
        Check if a symbol has been exported as defined here
            http://www.m4b.io/elf/export/binary/analysis/2015/05/25/what-is-an-elf-export.html
        :param symbol:
        :return:
        """
        symval, symndx, symbind, symtype = self.__get_symbol_info(symbol)
        # symbol bind in {STB_GLOBAL, STB_WEAK} type in {STT_OBJ, STT_FUNC, STT_LOOS}
        return symval > 0x0 and isinstance(symndx, int) and symbol.name and \
                        ('GLOBAL' in symbind or 'WEAK' in symbind) and \
                        ('OBJ' in symtype or 'FUNC' in symtype or 'LOOS' in symtype)

    def __show_dynamic_symbols(self, functions, ty):
        """
        Show symbols
        :param functions:
        :param ty:
        :return:
        """
        print ('%sed functions in %s: %d' % (ty, self.file_path, len(functions)))
        for f in sorted(functions.keys()):
            va = functions[f][1]
            print ("\t0x%010x %s" % (va, f))

    def read_imports(self, show=False):
        """
            The imported functions can be obtained from the following commands:
            (might not exactly match though)
            $ nm -DCg --defined-only [LIB_NAME.SO] | grep '^[0-9a-f]\+ [TtWwIiVv] '
            Here reads all dynamic symbols from the .dynsym section
        :param show:
        :return:
        """

        for section in self.elf.iter_sections():
            if isinstance(section, SymbolTableSection) and \
                            section['sh_entsize'] > 0 and C.DYN_SYMBOL_SECTION in section.name:
                for sym in section.iter_symbols():
                    if self._is_imported(sym):
                        self.imports[sym.name] = (None, sym['st_value'])

        if show:
            self.__show_dynamic_symbols(self.imports, ty='Import')

    def read_exports(self, show=False):
        """
        http://www.m4b.io/elf/export/binary/analysis/2015/05/25/what-is-an-elf-export.html
            The exports functions can be obtained from the following commands:
            $ nm -DCg [LIB_NAME.SO] | grep 'U '
        :param show:
        :return:
        """

        for section in self.elf.iter_sections():
            if isinstance(section, SymbolTableSection) and \
                            section['sh_entsize'] > 0 and C.DYN_SYMBOL_SECTION in section.name:
                for sym in section.iter_symbols():
                    if self._is_exported(sym):
                        self.exports[sym.name] = (self.file_path, sym['st_value'])

        if show:
            self.__show_dynamic_symbols(self.exports, ty='Export')

    @property
    def number_imports(self):
        return len(self.imports)

    @property
    def number_exports(self):
        return len(self.exports)

    def get_exports_lookup_by_fn(self):
        return self.exports_lookup_by_fn

    def get_exports_lookup_by_va(self):
        return self.exports_lookup_by_va

    def get_imports(self):
        return self.imports

    def get_exports(self):
        return self.exports

    def get_symbols(self):
        return self.symbols

    def read_shared_objects(self):
        """
        Read the needed shared objects for this ELF
        :return:
        """
        print ('Shared objects directly loaded by %s' % (self.file_path))
        # Collect the shared object needed by this ELF
        for section in self.elf.iter_sections():
            if not isinstance(section, DynamicSection):
                continue
            for tag in section.iter_tags():
                if tag.entry.d_tag == 'DT_NEEDED':
                    self.importedSO[tag.needed.split('.')[0]] = None
                    print('\tNeeded: [%s]' % tag.needed)
                elif tag.entry.d_tag == 'DT_RPATH':
                    print('\tLibrary rpath: [%s]' % tag.rpath)
                elif tag.entry.d_tag == 'DT_RUNPATH':
                    print('\tLibrary runpath: [%s]' % tag.runpath)
                elif tag.entry.d_tag == 'DT_SONAME':
                    print('\tLibrary soname: [%s]' % tag.soname)

    def get_section_va(self, sn):
        return self.section_ranges[sn][0]

    def getSectionByVA(self, va):
        secNames = self.section_ranges.keys()
        for sn in secNames:
            s, e = self.section_ranges[sn]
            if s <= va < e:
                return sn

    def is_within_section(self, kind, va):
        s, e = self.section_ranges[kind]
        return True if s <= va < e else False

def main():
    ''' This script can be working independently to offer ELF information'''
    import optparse

    usage = "Usage: %prog (-l|-s|-r|-o|-i|-e|-b) -f [target] (Use -h for help)"
    parser = optparse.OptionParser(usage=usage, version="%prog " + C.VERSION)

    parser.add_option("-l", "--headers", dest="hdr", action="store_true", default=False, help="show headers")
    parser.add_option("-s", "--sections", dest="sec", action="store_true", default=False, help="show sections")
    parser.add_option("-r", "--relocations", dest="reloc", action="store_true", default=False, help="show relocations)")
    parser.add_option("-o", "--sharedobjs", dest="so", action="store_true", default=False, help="show shared objects")
    parser.add_option("-i", "--imports", dest="imp", action="store_true", default=False, help="show imported functions")
    parser.add_option("-e", "--exports", dest="exp", action="store_true", default=False, help="show exported functions")
    parser.add_option("-b", "--symbols", dest="sym", action="store_true", default=False, help="show symbols")
    parser.add_option("-f", "--file", dest="exe", default=None, nargs=1, help="A target executable")

    (opts, args) = parser.parse_args()

    if not opts.exe:
        parser.error("A single target ELF must be provided!")

    f = opts.exe
    if os.path.exists(f):
        ep = ELFParser(f)
        if opts.hdr:
            ep.read_elf_headers()
        if opts.sec:
            ep.show_sections()
        if opts.reloc:
            ep.read_relocations()
        if opts.so:
            ep.read_shared_objects()
        if opts.imp:
            ep.read_imports(show=True)
        if opts.exp:
            ep.read_exports(show=True)
        if opts.sym:
            ep.read_symbols(show=True)

    else:
        parser.error("No such file exists!")

if __name__ == '__main__':
    main()
