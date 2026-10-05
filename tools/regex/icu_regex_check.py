#!/usr/bin/env python3
"""Compiles every pattern dumped by RegexDump.kt with the system ICU (libicui18n via ctypes).

Android's java.util.regex is backed by ICU, which is stricter than the desktop JDK (e.g. a bare
`}` or `{` outside a quantifier is a syntax error). A pattern that fails here throws
PatternSyntaxException inside a static initializer on the phone -> ExceptionInInitializerError,
then NoClassDefFoundError on every later use.

Usage: icu_regex_check.py patterns.tsv
"""
import ctypes
import ctypes.util
import glob
import re
import sys


def load_icu():
    libs = sorted(glob.glob("/usr/lib/*/libicui18n.so.*") + glob.glob("/usr/lib/libicui18n.so.*"))
    libs = [p for p in libs if re.search(r"\.so\.\d+$", p)]
    if not libs:
        found = ctypes.util.find_library("icui18n")
        if not found:
            sys.exit("ICU (libicui18n) not found")
        libs = [found]
    path = libs[-1]
    ver = re.search(r"\.so\.(\d+)", path).group(1)
    i18n = ctypes.CDLL(path)
    uc = ctypes.CDLL(path.replace("libicui18n", "libicuuc"))
    return i18n, uc, ver


class ParseError(ctypes.Structure):
    _fields_ = [("line", ctypes.c_int32), ("offset", ctypes.c_int32),
                ("pre", ctypes.c_uint16 * 16), ("post", ctypes.c_uint16 * 16)]


def main():
    i18n, uc, ver = load_icu()
    uopen = getattr(i18n, f"uregex_open_{ver}")
    uopen.restype = ctypes.c_void_p
    uopen.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32, ctypes.c_uint32,
                      ctypes.POINTER(ParseError), ctypes.POINTER(ctypes.c_int)]
    uclose = getattr(i18n, f"uregex_close_{ver}")
    uclose.argtypes = [ctypes.c_void_p]
    err_name = getattr(uc, f"u_errorName_{ver}")
    err_name.restype = ctypes.c_char_p

    total = bad = 0
    for line in open(sys.argv[1], encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        name, _flags, hexs = line.split("\t")
        units = [int(hexs[k:k + 4], 16) for k in range(0, len(hexs), 4)]
        buf = (ctypes.c_uint16 * max(1, len(units)))(*units)
        pe, status = ParseError(), ctypes.c_int(0)
        # ICU flags 0: on Android \w \s \d are Unicode-aware by default and
        # UNICODE_CHARACTER_CLASS is not passed through (see TextNormalizer.unicodePattern).
        handle = uopen(buf, len(units), 0, ctypes.byref(pe), ctypes.byref(status))
        total += 1
        if status.value > 0:
            bad += 1
            pat = "".join(map(chr, units))
            print(f"::error title=ICU regex::{name}: {err_name(status.value).decode()} at offset {pe.offset}: {pat}")
        elif handle:
            uclose(handle)
    print(f"ICU {ver}: {total - bad}/{total} patterns compile")
    if total == 0:
        sys.exit("no patterns found - dump step broken?")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
