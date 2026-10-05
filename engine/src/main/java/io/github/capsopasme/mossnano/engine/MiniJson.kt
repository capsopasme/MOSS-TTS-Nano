package io.github.capsopasme.mossnano.engine

/**
 * Tiny, dependency-free JSON reader. Kept platform-neutral (no org.json) so the
 * engine core can be unit-tested on a plain JVM.
 *
 * Objects -> LinkedHashMap<String, Any?>, arrays -> ArrayList<Any?>,
 * numbers -> Long or Double, plus String / Boolean / null.
 */
internal object MiniJson {
    fun parse(text: String): Any? {
        val p = Parser(text)
        p.skipWs()
        val value = p.readValue()
        p.skipWs()
        require(p.pos == text.length) { "Trailing data in JSON at ${p.pos}" }
        return value
    }

    private class Parser(val s: String) {
        var pos = 0

        fun skipWs() {
            while (pos < s.length) {
                val c = s[pos]
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') pos++ else break
            }
        }

        fun readValue(): Any? {
            skipWs()
            require(pos < s.length) { "Unexpected end of JSON" }
            return when (val c = s[pos]) {
                '{' -> readObject()
                '[' -> readArray()
                '"' -> readString()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                else -> if (c == '-' || c in '0'..'9') readNumber() else error("Bad JSON char '$c' at $pos")
            }
        }

        private fun literal(word: String, value: Any?): Any? {
            require(s.startsWith(word, pos)) { "Bad JSON literal at $pos" }
            pos += word.length
            return value
        }

        private fun readObject(): Map<String, Any?> {
            pos++ // {
            val map = LinkedHashMap<String, Any?>()
            skipWs()
            if (s[pos] == '}') { pos++; return map }
            while (true) {
                skipWs()
                val key = readString()
                skipWs()
                require(s[pos] == ':') { "Expected ':' at $pos" }
                pos++
                map[key] = readValue()
                skipWs()
                when (s[pos]) {
                    ',' -> pos++
                    '}' -> { pos++; return map }
                    else -> error("Expected ',' or '}' at $pos")
                }
            }
        }

        private fun readArray(): List<Any?> {
            pos++ // [
            val list = ArrayList<Any?>()
            skipWs()
            if (s[pos] == ']') { pos++; return list }
            while (true) {
                list.add(readValue())
                skipWs()
                when (s[pos]) {
                    ',' -> pos++
                    ']' -> { pos++; return list }
                    else -> error("Expected ',' or ']' at $pos")
                }
            }
        }

        private fun readString(): String {
            require(s[pos] == '"') { "Expected string at $pos" }
            pos++
            val sb = StringBuilder()
            while (true) {
                val c = s[pos++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        when (val e = s[pos++]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                sb.append(s.substring(pos, pos + 4).toInt(16).toChar())
                                pos += 4
                            }
                            else -> error("Bad escape '\\$e' at $pos")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun readNumber(): Any {
            val start = pos
            if (s[pos] == '-') pos++
            var isDouble = false
            while (pos < s.length) {
                val c = s[pos]
                if (c in '0'..'9') {
                    pos++
                } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                    isDouble = true
                    pos++
                } else {
                    break
                }
            }
            val token = s.substring(start, pos)
            return if (isDouble) token.toDouble() else token.toLongOrNull() ?: token.toDouble()
        }
    }
}

@Suppress("UNCHECKED_CAST")
internal fun Any?.obj(): Map<String, Any?> = this as? Map<String, Any?> ?: error("JSON object expected, got $this")

@Suppress("UNCHECKED_CAST")
internal fun Any?.arr(): List<Any?> = this as? List<Any?> ?: error("JSON array expected")

internal fun Any?.int(): Int = (this as? Number)?.toInt() ?: error("JSON number expected, got $this")

internal fun Any?.dbl(): Double = (this as? Number)?.toDouble() ?: error("JSON number expected, got $this")

internal fun Any?.str(): String = this as? String ?: error("JSON string expected, got $this")

internal fun Any?.intArray(): IntArray {
    val list = arr()
    return IntArray(list.size) { list[it].int() }
}

internal fun Any?.longArray(): LongArray {
    val list = arr()
    return LongArray(list.size) { (list[it] as Number).toLong() }
}
