import java.io.File
import java.util.regex.Pattern

/**
 * Loads every compiled class under args[0] and prints each static Regex / Pattern field as
 * "<Class>.<field>\t<javaFlags>\t<pattern as UTF-16 hex>" so icu_regex_check.py can compile the
 * exact strings with ICU — the regex engine behind java.util.regex on Android.
 * Kotlin `object` properties compile to static fields, so this covers all of them.
 */
fun main(args: Array<String>) {
    val root = File(args[0])
    val loader = java.net.URLClassLoader(arrayOf(root.toURI().toURL()), Thread.currentThread().contextClassLoader)
    root.walk().filter { it.isFile && it.name.endsWith(".class") }.sortedBy { it.path }.forEach { f ->
        val name = f.relativeTo(root).path.removeSuffix(".class").replace(File.separatorChar, '.')
        if (name == "RegexDumpKt") return@forEach
        val cls = Class.forName(name, true, loader)  // init failure here = JVM-side bug, fail loudly
        for (field in cls.declaredFields) {
            if (!java.lang.reflect.Modifier.isStatic(field.modifiers)) continue
            field.isAccessible = true
            val (p, flags) = when (val v = field.get(null)) {
                is Regex -> v.toPattern().let { it.pattern() to it.flags() }
                is Pattern -> v.pattern() to v.flags()
                else -> continue
            }
            val hex = p.map { String.format("%04x", it.code) }.joinToString("")
            println("${cls.name.substringAfterLast('.')}.${field.name}\t$flags\t$hex")
        }
    }
}
