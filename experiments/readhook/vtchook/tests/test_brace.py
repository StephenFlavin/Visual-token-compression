"""Brace/semicolon family (vtchook/minify/brace.py): invariant, token-stream round trip, edge cases per language.

No network, no Claude calls. The snippets imitate real code; the real-corpus checks (56 popular files, also compared
against go/scanner, the TypeScript parser, PHP's tokenizer, clang's raw lexer, javac's and rustc's parse trees) were
run when the module was written.
"""
import os
import re
import sys
import time

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(HERE))
from vtchook.minify import REGISTRY, detect, minify, preserves  # noqa: E402
from vtchook.minify import brace  # noqa: E402
from vtchook.minify.base import strip_marks  # noqa: E402

EXT = {"javascript": "js", "typescript": "ts", "tsx": "tsx", "java": "java", "kotlin": "kt", "scala": "scala",
       "c": "c", "cpp": "cpp", "csharp": "cs", "go": "go", "rust": "rs", "swift": "swift", "php": "php",
       "dart": "dart"}


def run(kind, src):
    """Minify through the public entry point and check everything that must hold for any input."""
    r = minify(src, f"x.{EXT[kind]}")
    assert r is not None, "minifier refused or broke the invariant"
    flow, m = r
    assert m.kind == kind
    assert preserves(m, src, flow)
    assert brace.flow_tokens_match(src, flow, kind), "token stream changed"
    lines = src.split("\n")
    for t in re.finditer(r"§(\d+) (\S)", flow):                  # every tag sits on its real line
        n, ch = int(t.group(1)), t.group(2)
        assert lines[n - 1].strip().startswith(ch), (n, lines[n - 1], flow[t.start():t.start() + 40])
    return flow


def test_registration():
    for kind, ext in EXT.items():
        assert kind in REGISTRY
        assert detect(f"/a/b/file.{ext}").kind == kind
        assert 1.5 < REGISTRY[kind].cpt < 4
    assert detect("x.h").kind == "c" and detect("x.hpp").kind == "cpp" and detect("x.jsx").kind == "javascript"
    assert detect("x.mts").kind == "typescript" and detect("x.kts").kind == "kotlin"


# ------------------------------------------------------------------------------------------------ JS / TS / TSX

def test_js_spacing_and_operators():
    src = "function f(a, b) {\n    const x = a - -b;\n    let y = a + +b, z = a / b / 2;\n    return x < -1 && y > 2;\n}\n"
    flow = run("javascript", src)
    assert "§1 function f(a,b){ §2 const x=a- -b; §3 let y=a+ +b,z=a/b/2; §4 return x< -1&&y>2;}" == flow


def test_js_regex_vs_division_and_template():
    src = ("const re = /[/ ]+  x/g, half = total / 2 / n;\n"
           "if (s.match(/ab  c/)) { out = `a  ${ cond ? `in  ${x}` : '  }' }  b`; }\n"
           "const q = a.b / c, r = (x) / 2;\n")
    flow = run("javascript", src)
    assert "/[/ ]+  x/g" in flow and "/ab  c/" in flow
    assert "`a  ${ cond ? `in  ${x}` : '  }' }  b`" in flow
    assert "half=total/2/n" in flow and "q=a.b/c,r=(x)/2" in flow


def test_js_asi_lines_keep_their_tags():
    src = "function f() {\n  return\n  x\n}\na\n++b\nconst s = 'a' +\n  'b'\nfoo()\n  .bar()\n  .baz()\n"
    flow = run("javascript", src)
    assert "return §3 x}" in flow           # `return\nx` returns undefined: the newline (tag) must stay
    assert "a §6 ++b" in flow
    assert "const s='a'+'b' §9 foo().bar().baz()" in flow


def test_jsx():
    src = ("export default function App({ items }) {\n"
           "  return (\n"
           "    <div className=\"app  main\" onClick={() => go(1)}>\n"
           "      Don't   stop {items.length > 0 && <List items={items} />}\n"
           "      <>frag</>\n"
           "    </div>\n"
           "  );\n"
           "}\n"
           "const isLess = a < b && c > d;\n")
    flow = run("javascript", src)
    assert "\"app  main\"" in flow and "Don't stop" in flow and "<List items={items} />" in flow
    assert "isLess=a<b&&c>d" in flow


def test_tsx_generic_arrow_and_types():
    src = ("const id = <T,>(x: T): T => x;\n"
           "type M = Map<string, Array<number> >;\n"
           "function g<T extends object>(x: T) { return <span title=\"it's\">{x} </span>; }\n"
           "const h = <U extends Base>(u: U) => u;\n"
           "}\n/ a +b/.test(s) && go();\n")
    flow = run("tsx", src)
    assert "id= <T,>(x:T):T=>x;" in flow and "<span title=\"it's\">{x} </span>" in flow
    assert "h= <U extends Base>(u:U)=>u;" in flow
    assert "} §6 / a +b/.test(s)&&go();" in flow     # a regex at a line start after a block
    flow = run("typescript", src.replace("return <span title=\"it's\">{x} </span>;", "return x;"))
    assert "Map<string,Array<number> >" in flow and "g<T extends object>(x:T){return x;}" in flow


def test_ts_decorators_and_optional_chaining():
    src = "@Component({\n  selector: 'app',\n})\nexport class A {\n  @Input() name?: string;\n  f() { return this.x?.y ?? 0; }\n}\n"
    flow = run("typescript", src)
    assert "§1 @Component({ §2 selector:'app',}) §4 export class A{ §5 @Input()name?:string; §6 f(){return this.x?.y??0;}}" == flow


def test_unterminated_string_falls_back():
    assert minify("const s = 'abc\nfoo();\n", "x.js") is None
    assert minify("int x; /* never closed\n", "x.c") is None
    assert minify("x = 1; // §5 tag char\n", "x.js") is None   # marker characters in the source


# ------------------------------------------------------------------------------------------------ Java / Kotlin / Scala

def test_java_text_block_annotations_chains():
    src = ('package a;\n\npublic class A {\n    @Override\n    public String toString() {\n'
           '        String s = """\n            Hello,   "world"\n              indented\n            """;\n'
           '        return list.stream()\n            .map(x -> x + 1)\n            .collect(toList());\n'
           '    }\n    char c = \'"\', d = \'\\\'\';\n    Map<String, List<Integer>> m;\n}\n')
    flow = run("java", src)
    assert '"""\n            Hello,   "world"\n              indented\n            """' in flow
    assert "§4 @Override §5 public String toString(){" in flow
    assert "return list.stream().map(x->x+1).collect(toList());}" in flow
    assert "char c='\"',d='\\'';" in flow


def test_kotlin_strings_comments_and_operators():
    src = ('/* outer /* nested */ still comment */\n'
           'fun `my test name`() {\n'
           '    val s = """\n        |a  ${"}" + x}\n    """"\n'
           '    val t = "v=${map["k  "]}"\n'
           '    if (x !is String && y !in set) return\n'
           '    val z = a ?: b\n'
           '        ?.let { it }\n'
           '}\n')
    flow = run("kotlin", src)
    assert "/* outer /* nested */ still comment */" in flow
    assert "fun `my test name`(){" in flow
    assert '"""\n        |a  ${"}" + x}\n    """"' in flow and '"v=${map["k  "]}"' in flow
    assert "x !is String&&y !in set" in flow
    assert "val z=a?:b?.let{it}" in flow


def test_scala_interpolation_symbols_triple():
    src = ('object A {\n  val s = s"${x + "}"}  y"\n  val r = raw"""a  \\d"""\n  val sym = \'foo\n'
           '  val c = \'c\'\n  xs.map { case (a, b) => a + -b }\n  /* a /* b */ c */\n}\n')
    flow = run("scala", src)
    assert 's"${x + "}"}  y"' in flow and 'raw"""a  \\d"""' in flow
    assert "val sym='foo" in flow and "val c='c'" in flow
    assert "=>a+ -b" in flow


# ------------------------------------------------------------------------------------------------ C / C++ / C#

def test_c_preprocessor():
    src = ("#include <stdio.h>\n#define F (x)\n#define G(a, b) \\\n    ((a) +  (b))\n"
           "#error don't do this\n#if defined(X) // comment\nint y;\n#endif\n"
           "int f(int *p) { return a / *p + b; }  // trailing\nint g;\n")
    flow = run("c", src)
    assert "§1 #include <stdio.h> §2 #define F (x) §3 #define G(a, b) \\ ((a) + (b)) §5 #error don't do this" in flow
    assert "§6 #if defined(X) // comment §7 int y; §8 #endif §9 int f(int*p){return a/ *p+b;} // trailing §10 int g;" \
        in flow


def test_cpp_raw_strings_digit_separators_templates():
    src = ('auto s = R"x(a "quoted" )" string\n   next  line)x";\n'
           'long n = 1\'000\'000;\nchar c = \'\\n\';\n'
           'std::vector<std::vector<int> > v;\nstd::cout << "x"\n    << y;\n'
           'template <typename T>\nT max(T a, T b) { return a > b ? a : b; }\n')
    flow = run("cpp", src)
    assert 'R"x(a "quoted" )" string\n   next  line)x"' in flow
    assert "n=1'000'000;" in flow and "std::vector<std::vector<int> >v;" in flow
    assert 'std::cout<<"x"<<y;' in flow
    assert "§8 template<typename T> §9 T max(T a,T b){return a>b?a:b;}" in flow


def test_csharp_strings_and_directives():
    src = ('#region Don\'t touch\nclass A\n{\n    string v = @"C:\\dir  ""q""\n  next";\n'
           '    string i = $"{(x ? "a" : "b")}  {{lit}} {y:N2}";\n'
           '    string r = """\n        { "a":  1 }\n        """;\n'
           '    string vi = $@"{a}  ""b""";\n    int @class = 1;\n}\n#endregion\n')
    flow = run("csharp", src)
    assert '@"C:\\dir  ""q""\n  next"' in flow and '$"{(x ? "a" : "b")}  {{lit}} {y:N2}"' in flow
    assert '"""\n        { "a":  1 }\n        """' in flow and '$@"{a}  ""b"""' in flow
    assert "§1 #region Don't touch §2 class A{" in flow     # Allman brace joined to its header
    assert "int @class=1;} §13 #endregion" in flow


# ------------------------------------------------------------------------------------------------ Go / Rust / Swift

def test_go_raw_strings_and_newlines():
    src = ("package main\n\nimport (\n\t\"fmt\"\n)\n\nconst q = `raw\n  text`\n\n"
           "func main() {\n\tr := '\\''\n\tx := a +\n\t\tb\n\tif x < -y {\n\t}\n\tfmt.Println(x, r)\n}\n")
    flow = run("go", src)
    assert "const q=`raw\n  text`" in flow
    assert "§12 x:=a+b §14 if x< -y{} §16 fmt.Println(x,r)}" in flow   # Go's auto-semicolons sit at tags


def test_rust_raw_strings_lifetimes_nested_comments():
    src = ('/* a /* b */ c */\nfn f<\'a>(s: &\'a str) -> &\'a str {\n    let r = r#"a "b"  c"#;\n'
           '    let b = br##"x"#y"##;\n    let c = b\'x\';\n    let q = \'"\';\n    \'outer: loop { break \'outer; }\n'
           '    let v = a\n        + b;\n    s\n}\n#[derive(Debug)]\nstruct S;\n')
    flow = run("rust", src)
    assert "/* a /* b */ c */" in flow and 'r#"a "b"  c"#' in flow and 'br##"x"#y"##' in flow
    assert "fn f<'a>(s: &'a str)-> &'a str{" in flow
    assert "'outer:loop{break 'outer;}" in flow and "let v=a+b;" in flow


def test_swift_balanced_operators_and_strings():
    src = ('#if DEBUG\nlet a = b ? c : d\n#endif\nlet x = -1\nlet y = a + b\n'
           'let s = "v: \\(f("a  b")) end"\nlet r = #"raw "x" \\(y)"#\n'
           'let m = """\n    multi  \\(z)\n    """\nfunc g(_ x: Int) -> Int? { return x }\n'
           'let z = list\n    .filter { $0 > 1 }\n')
    flow = run("swift", src)
    assert "§1 #if DEBUG §2 let a=b ? c:d §3 #endif" in flow     # `?` keeps its spaces in Swift
    assert "let x= -1" not in flow and "let x = -1" in flow       # `=` cannot lose only one side
    assert "let y=a+b" in flow
    assert '"v: \\(f("a  b")) end"' in flow and '#"raw "x" \\(y)"#' in flow
    assert "func g(_ x:Int)->Int? {return x}" in flow
    assert "let z=list .filter{$0>1}" in flow


# ------------------------------------------------------------------------------------------------ PHP / Dart

def test_php_modes_and_strings():
    src = ('<html>\n  <body>\n<?php\n#[Attribute]\nclass A {}\n$a = "x {$b["k  "]} y";\n'
           '$h = <<<EOT\n  keep   this\n    EOT;\n$n = <<<\'NOW\'\nraw $x\nNOW;\n'
           '# hash comment\n$x = $y ?-> z;\n// comment ?> <p>  html  text</p>\n<?= $v ?>\n</body>\n')
    flow = run("php", src)
    assert flow.startswith("§1 <html> §2 <body> §3 <?php §4 #[Attribute] §5 class A{}")
    assert '"x {$b["k  "]} y"' in flow and "<<<EOT\n  keep   this\n    EOT" in flow
    assert "<<<'NOW'\nraw $x\nNOW" in flow
    assert "// comment ?> <p> html text</p> §16 <?= $v ?>" in flow


def test_dart_strings():
    src = ("void main() {\n  var r = r'a\\n  b';\n  var t = '''x ${y}\n   z''';\n"
           "  var u = \"${a[\"b  \"]}\";\n  /* a /* b */ c */\n  print(r + t);\n}\n")
    flow = run("dart", src)
    assert "r'a\\n  b'" in flow and "'''x ${y}\n   z'''" in flow and '"${a["b  "]}"' in flow


# ------------------------------------------------------------------------------------------------ tags, policies, speed

def test_comment_runs_and_forced_tags():
    src = "int a; // one\n// two\n// three\nint b = f(x,\n    y);\n/**\n * Doc   text\n */\nint c;\n"
    flow = run("java", src)
    assert flow == "§1 int a; // one // two // three §4 int b=f(x,y); §6 /** * Doc text */ §9 int c;"


def test_untagged_runs_are_capped():
    src = "int[] a = {\n" + "".join(f"  {i},\n" for i in range(30)) + "};\n"
    flow = run("java", src)
    tags = [int(n) for n in re.findall(r"§(\d+) ", flow)]
    assert max(b - a for a, b in zip(tags, tags[1:])) <= brace.MAX_UNTAGGED + 1
    assert len(tags) < 8


def test_policies():
    src = "class A {\n  void f() {\n    g(1,\n      2);\n  }\n}\n"
    assert brace.brace_flow(src, "java", policy="all") == "§1 class A{ §2 void f(){ §3 g(1, §4 2); §5 } §6 }"
    assert brace.brace_flow(src, "java") == "§1 class A{ §2 void f(){ §3 g(1,2);}}"
    assert brace.brace_flow(src, "java", policy="first") == "§1 class A{void f(){g(1,2);}}"


def test_whitespace_only_changes_and_speed():
    body = "".join(f"    if (x{i} > {i}) {{ total += compute(\"s  {i}\", x{i}); }} // note {i}\n" for i in range(4000))
    src = "public class Big {\n  int f() {\n" + body + "  }\n}\n"
    t = time.time()
    flow = run("java", src)
    assert time.time() - t < 10
    assert len(flow) < len(src)
    assert re.sub(r"\s", "", strip_marks(flow)) == re.sub(r"\s", "", src)


@pytest.mark.parametrize("kind", sorted(EXT))
def test_empty_and_trivial(kind):
    assert minify("", f"x.{EXT[kind]}") is None
    assert run(kind, "x\n") == "§1 x"
