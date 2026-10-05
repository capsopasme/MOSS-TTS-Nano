package io.github.capsopasme.mossnano.engine

import java.util.regex.Matcher
import java.util.regex.Pattern

/**
 * Text normalization before tokenization.
 *
 * Mirrors the official pipeline (text_normalization_pipeline.prepare_tts_request_texts):
 *   robust cleanup  ->  semantic TN (numbers, dates, %, units…)  ->  robust cleanup
 *
 * The robust stage is a line-by-line port of tts_robust_normalizer_single_script.py.
 * The semantic stage replaces WeTextProcessing (an FST stack that is far too heavy for a
 * phone) with a compact rule set covering what actually shows up in chat/reading text.
 */
object TextNormalizer {

    fun normalize(text: String): String {
        var t = Robust.normalize(text)
        if (t.isEmpty()) return t
        t = if (TextChunker.containsCjk(t)) ZhTn.normalize(t) else EnTn.normalize(t)
        return Robust.normalize(t)
    }

    // =====================================================================================
    // Robust (non-semantic) cleanup — port of the official single-script normalizer
    // =====================================================================================
    internal object Robust {
        private const val CJK_CHARS = "\\u3400-\\u4dbf\\u4e00-\\u9fff\\u3040-\\u30ff"
        private const val CJK = "[$CJK_CHARS]"
        private const val PROT = "___PROT\\d+___"
        private const val LATINISH = "(?:$PROT|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"
        private const val U = Pattern.UNICODE_CHARACTER_CLASS

        private val URL = Pattern.compile("https?://[^\\s\\u3000，。！？；、）】》〉」』]+", U)
        private val EMAIL = Pattern.compile("(?<![\\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}(?![\\w.-])", U)
        private val MENTION = Pattern.compile("(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")
        private val REDDIT = Pattern.compile("(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")
        private val HASHTAG = Pattern.compile("(?<![A-Za-z0-9_])#(?!\\s)[^\\s#]+", U)
        private val DOT_TOKEN = Pattern.compile("(?<![A-Za-z0-9_])\\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+")
        private val FILELIKE = Pattern.compile(
            "(?<![A-Za-z0-9_])(?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[./+:-])[A-Za-z0-9][A-Za-z0-9._/+:-]*(?![A-Za-z0-9_])"
        )
        val PROTECT_PATTERNS = listOf(URL, EMAIL, MENTION, REDDIT, HASHTAG, DOT_TOKEN, FILELIKE)
        private val PROT_RE = Pattern.compile(PROT)
        private val ZERO_WIDTH = Regex("[\\u200b-\\u200d\\ufeff]")
        private const val TRAILING_CLOSERS = "\"')]}）】》〉」』”’"
        private val FLOW_ARROWS = Regex("\\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\\s*")

        fun normalize(input: String): String {
            var text = baseCleanup(input)
            text = markdownAndLines(text)
            text = FLOW_ARROWS.replace(text, "，")
            val protected = ArrayList<String>()
            text = protect(text, protected)
            text = visibleUnderscores(text)
            text = spaces(text)
            text = structuralPunctuation(text)
            text = repeatedPunctuation(text)
            text = spaces(text)
            text = restore(text, protected)
            text = text.trim()
            return terminalByLine(text)
        }

        fun protect(input: String, protected: MutableList<String>): String {
            var text = input
            for (p in PROTECT_PATTERNS) {
                val m = p.matcher(text)
                val sb = StringBuffer()
                while (m.find()) {
                    val idx = protected.size
                    protected += m.group()
                    m.appendReplacement(sb, Matcher.quoteReplacement("___PROT${idx}___"))
                }
                m.appendTail(sb)
                text = sb.toString()
            }
            return text
        }

        fun restore(input: String, protected: List<String>): String {
            var text = input
            // restore in reverse so ___PROT1___ doesn't clobber ___PROT10___
            for (idx in protected.indices.reversed()) text = text.replace("___PROT${idx}___", protected[idx])
            return text
        }

        private fun baseCleanup(input: String): String {
            val text = ZERO_WIDTH.replace(
                input.replace("\r\n", "\n").replace("\r", "\n").replace("　", " "), ""
            )
            val sb = StringBuilder(text.length)
            for (ch in text) {
                if (ch == '\n' || ch == '\t' || ch == ' ' || !isControlCategory(ch)) sb.append(ch)
            }
            return sb.toString()
        }

        private fun isControlCategory(ch: Char): Boolean = when (Character.getType(ch).toByte()) {
            Character.CONTROL, Character.FORMAT, Character.PRIVATE_USE, Character.SURROGATE, Character.UNASSIGNED -> true
            else -> false
        }

        private val MD_LINK = Regex("\\[([^\\[\\]]+?)]\\((https?://[^)\\s]+)\\)")
        private val MD_HEADING = Regex("^#{1,6}\\s+")
        private val MD_QUOTE = Regex("^>\\s+")
        private val MD_BULLET = Regex("^[-*+]\\s+")
        private val MD_ORDERED = Regex("^\\d+[.)]\\s+")

        private fun markdownAndLines(input: String): String {
            val text = MD_LINK.replace(input, "$1 $2")
            val lines = text.lines().mapNotNull { raw ->
                var line = raw.trim()
                if (line.isEmpty()) return@mapNotNull null
                line = MD_HEADING.replace(line, "")
                line = MD_QUOTE.replace(line, "")
                line = MD_BULLET.replace(line, "")
                line = MD_ORDERED.replace(line, "")
                line
            }
            if (lines.isEmpty()) return ""
            val sb = StringBuilder()
            for ((i, line) in lines.withIndex()) {
                sb.append(if (i < lines.lastIndex) ensureTerminal(line) else line)
            }
            return sb.toString()
        }

        private fun visibleUnderscores(text: String): String {
            val sb = StringBuilder(text.length)
            val m = PROT_RE.matcher(text)
            var last = 0
            while (m.find()) {
                sb.append(text.substring(last, m.start()).replace('_', ' ')).append(m.group())
                last = m.end()
            }
            sb.append(text.substring(last).replace('_', ' '))
            return sb.toString()
        }

        private val WS = Regex("[ \\t\\r\\f\\u000B]+")
        private val CJK_SPACE_CJK = Regex("($CJK)\\s+(?=$CJK)")
        private val CJK_SPACE_DIGIT = Regex("($CJK)\\s+(?=\\d)")
        private val DIGIT_SPACE_CJK = Regex("(\\d)\\s+(?=$CJK)")
        private val CJK_BEFORE_LATIN = Regex("($CJK)(?=($LATINISH))")
        private val LATIN_BEFORE_CJK = Regex("(($LATINISH))(?=$CJK)")
        private val MULTI_SPACE = Regex(" {2,}")
        private val SPACE_BEFORE_ZH_CLOSE = Regex("\\s+([，。！？；：、”’」』】）》])")
        private val SPACE_AFTER_ZH_OPEN = Regex("([（【「『《“‘])\\s+")
        private val SPACE_AFTER_ZH_PUNCT = Regex("([，。！？；：、])\\s*")
        private val SPACE_BEFORE_ASCII_PUNCT = Regex("\\s+([,.;!?])")

        private fun spaces(input: String): String {
            var t = WS.replace(input, " ")
            t = CJK_SPACE_CJK.replace(t, "$1")
            t = CJK_SPACE_DIGIT.replace(t, "$1")
            t = DIGIT_SPACE_CJK.replace(t, "$1")
            t = CJK_BEFORE_LATIN.replace(t, "$1 ")
            t = LATIN_BEFORE_CJK.replace(t, "$1 ")
            t = MULTI_SPACE.replace(t, " ")
            t = SPACE_BEFORE_ZH_CLOSE.replace(t, "$1")
            t = SPACE_AFTER_ZH_OPEN.replace(t, "$1")
            t = SPACE_AFTER_ZH_PUNCT.replace(t, "$1")
            t = SPACE_BEFORE_ASCII_PUNCT.replace(t, "$1")
            return MULTI_SPACE.replace(t, " ").trim()
        }

        private val SQUARE = Regex("\\[\\s*([^\\[\\]]+?)\\s*]")
        private val CURLY = Regex("\\{\\s*([^{}]+?)\\s*}")
        private val CJK_BRACKETS = Regex("[【〖『「]\\s*([^】〗』」]+?)\\s*[】〗』」]")
        private val STANDALONE_TITLE =
            Regex("(^|[。！？!?；;]\\s*)《([^》]+)》(?=\\s*(?:___PROT\\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))")
        private val LONG_DASH = Regex("\\s*(?:—|–|―|-){2,}\\s*")

        private fun structuralPunctuation(input: String): String {
            var t = SQUARE.replace(input, "\"$1\"")
            t = CURLY.replace(t, "\"$1\"")
            t = CJK_BRACKETS.replace(t, "\"$1\"")
            t = STANDALONE_TITLE.replace(t, "$1$2")
            t = FLOW_ARROWS.replace(t, "，")
            t = LONG_DASH.replace(t, "。")
            return t
        }

        private val ELLIPSIS = Regex("(?:\\.{3,}|…{2,}|……+)")
        private val REP_PERIOD = Regex("[。．]{2,}")
        private val REP_COMMA = Regex("[，,]{2,}")
        private val REP_EXCL = Regex("[!！]{2,}")
        private val REP_Q = Regex("[?？]{2,}")
        private val REP_MIXED = Regex("[!?！？]{2,}")

        private fun repeatedPunctuation(input: String): String {
            var t = ELLIPSIS.replace(input, "。")
            t = REP_PERIOD.replace(t, "。")
            t = REP_COMMA.replace(t, "，")
            t = REP_EXCL.replace(t, "！")
            t = REP_Q.replace(t, "？")
            t = REP_MIXED.replace(t) { m ->
                val s = m.value
                val hasQ = s.any { it == '?' || it == '？' }
                val hasE = s.any { it == '!' || it == '！' }
                if (hasQ && hasE) "？！" else if (hasQ) "？" else "！"
            }
            return t
        }

        private fun isPunctuation(ch: Char): Boolean = when (Character.getType(ch).toByte()) {
            Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
            Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION -> true
            else -> false
        }

        fun ensureTerminal(text: String): String {
            if (text.isEmpty()) return text
            var i = text.length - 1
            while (i >= 0 && text[i].isWhitespace()) i--
            while (i >= 0 && text[i] in TRAILING_CLOSERS) i--
            return if (i >= 0 && isPunctuation(text[i])) text else "$text。"
        }

        private fun terminalByLine(text: String): String {
            if (text.isEmpty()) return text
            return text.split("\n").joinToString("\n") { line ->
                val s = line.trim()
                if (s.isEmpty()) "" else ensureTerminal(s)
            }.trim()
        }
    }

    /** Protects URLs/emails/file names etc. with digit-free placeholders so number rules can't touch them. */
    internal fun protectForTn(input: String, protected: MutableList<String>): String {
        var text = input
        for (p in Robust.PROTECT_PATTERNS) {
            val m = p.matcher(text)
            val sb = StringBuffer()
            while (m.find()) {
                val key = tnKey(protected.size)
                protected += m.group()
                m.appendReplacement(sb, Matcher.quoteReplacement(key))
            }
            m.appendTail(sb)
            text = sb.toString()
        }
        return text
    }

    internal fun restoreForTn(input: String, protected: List<String>): String {
        var text = input
        for (idx in protected.indices) text = text.replace(tnKey(idx), protected[idx])
        return text
    }

    private fun tnKey(index: Int): String {
        val sb = StringBuilder("\uE000")
        var n = index
        do {
            sb.append('a' + n % 26)
            n /= 26
        } while (n > 0)
        return sb.append('\uE001').toString()
    }

    // =====================================================================================
    // Chinese semantic TN (numbers / dates / time / percent / currency / units)
    // =====================================================================================
    internal object ZhTn {
        private val DIGITS = charArrayOf('零', '一', '二', '三', '四', '五', '六', '七', '八', '九')
        private const val MEASURE = "个只本件条张位台次天年月周岁斤块元头匹辆架间座把双对份种名"

        fun digitsOf(s: String): String = buildString { for (c in s) if (c in '0'..'9') append(DIGITS[c - '0']) }

        /** 0..9999 */
        private fun section(n: Int, leadingZero: Boolean): String {
            val units = arrayOf("", "十", "百", "千")
            val sb = StringBuilder()
            var zero = false
            var started = false
            for (pos in 3 downTo 0) {
                val d = (n / pow10(pos)) % 10
                if (d == 0) {
                    if (started || leadingZero) zero = true
                } else {
                    if (zero && (started || leadingZero)) sb.append('零')
                    zero = false
                    sb.append(DIGITS[d]).append(units[pos])
                    started = true
                }
            }
            return sb.toString()
        }

        private fun pow10(p: Int): Int = when (p) { 0 -> 1; 1 -> 10; 2 -> 100; else -> 1000 }

        fun cardinal(numStr: String): String {
            val s = numStr.trimStart('0')
            if (s.isEmpty()) return "零"
            if (s.length > 16) return digitsOf(numStr)
            var n = s.toLong()
            if (n == 0L) return "零"
            val bigUnits = arrayOf("", "万", "亿", "万亿")
            val parts = ArrayList<Int>()
            while (n > 0) {
                parts += (n % 10000).toInt()
                n /= 10000
            }
            val sb = StringBuilder()
            var needZero = false
            for (i in parts.indices.reversed()) {
                val part = parts[i]
                if (part == 0) {
                    needZero = sb.isNotEmpty()
                    continue
                }
                val leading = sb.isNotEmpty() && (needZero || part < 1000)
                sb.append(section(part, leading)).append(bigUnits[i])
                needZero = false
            }
            var out = sb.toString()
            if (out.startsWith("一十")) out = out.substring(1)  // 十五 rather than 一十五
            return out
        }

        fun decimal(intPart: String, frac: String?): String {
            val i = cardinal(intPart)
            return if (frac.isNullOrEmpty()) i else i + "点" + digitsOf(frac)
        }

        private fun numberWord(intPart: String, frac: String?): String {
            if (frac == null && intPart.length > 1 && intPart.startsWith("0")) return digitsOf(intPart)
            if (frac == null && intPart.length >= 11) return digitsOf(intPart) // phone / ids
            return decimal(intPart, frac)
        }

        private val THOUSANDS = Regex("(?<![\\d.])(\\d{1,3}(?:,\\d{3})+)(?![\\d,])")
        private val DATE_YMD = Regex("(?<!\\d)(\\d{4})[-/.年](\\d{1,2})[-/.月](\\d{1,2})日?(?!\\d)")
        private val DATE_YM = Regex("(?<!\\d)(\\d{4})[-/](\\d{1,2})(?![\\d.]|-\\d)")
        private val YEAR = Regex("(?<!\\d)(\\d{4})(?=年)")
        private val TIME = Regex("(?<!\\d)([01]?\\d|2[0-4])[:：]([0-5]\\d)(?:[:：]([0-5]\\d))?(?!\\d)")
        private val PERCENT = Regex("(-)?(\\d+)(?:\\.(\\d+))?\\s*[%％]")
        private val PERMILLE = Regex("(-)?(\\d+)(?:\\.(\\d+))?\\s*‰")
        private val CURRENCY_PRE = Regex("([¥￥$€£])\\s*(\\d+(?:\\.\\d+)?)")
        private val RANGE = Regex("(\\d+(?:\\.\\d+)?)\\s*[-~～—–]\\s*(?=\\d)")
        private val NEGATIVE = Regex("(^|[\\s=:+*/,(，：；;（【\\[{\\u3400-\\u9fff])-(?=\\d)")
        private val FRACTION = Regex("(?<![\\d/])(\\d+)/(\\d+)(?![\\d/])")
        private val TEMP = Regex("(\\d+(?:\\.\\d+)?)\\s*(?:°C|℃)")
        private val UNIT = Regex("(\\d)\\s*(km/h|km|kg|cm|mm|ml|mL|m²|㎡|m³|kW|°)(?![A-Za-z])")
        private val NUMBER = Regex("(\\d+)(?:\\.(\\d+))?")
        private val CJK_HYPHEN_CJK = Regex("([\\u3400-\\u9fff])\\s*-\\s*(?=[\\u3400-\\u9fff])")
        private val OTHER_HYPHEN = Regex("([^\\s\\d-])\\s*-\\s*(?=[^\\s\\d-])")

        private val UNIT_WORDS = mapOf(
            "km/h" to "公里每小时", "km" to "公里", "kg" to "公斤", "cm" to "厘米", "mm" to "毫米",
            "ml" to "毫升", "mL" to "毫升", "m²" to "平方米", "㎡" to "平方米", "m³" to "立方米",
            "kW" to "千瓦", "°" to "度",
        )
        private val CURRENCY_WORDS = mapOf("¥" to "元", "￥" to "元", "$" to "美元", "€" to "欧元", "£" to "英镑")

        fun normalize(input: String): String {
            val protected = ArrayList<String>()
            var t = protectForTn(input, protected)

            t = THOUSANDS.replace(t) { it.value.replace(",", "") }
            t = DATE_YMD.replace(t) { m ->
                val mo = m.groupValues[2].toInt()
                val d = m.groupValues[3].toInt()
                if (mo in 1..12 && d in 1..31) digitsOf(m.groupValues[1]) + "年" + cardinal(mo.toString()) + "月" + cardinal(d.toString()) + "日"
                else m.value
            }
            t = DATE_YM.replace(t) { m ->
                val mo = m.groupValues[2].toInt()
                if (mo in 1..12) digitsOf(m.groupValues[1]) + "年" + cardinal(mo.toString()) + "月" else m.value
            }
            t = YEAR.replace(t) { digitsOf(it.groupValues[1]) }
            t = TIME.replace(t) { m ->
                val h = m.groupValues[1].toInt()
                val mi = m.groupValues[2].toInt()
                val sec = m.groupValues[3]
                buildString {
                    append(if (h == 2) "两" else cardinal(h.toString())).append("点")
                    if (mi != 0 || sec.isNotEmpty()) {
                        if (mi in 1..9) append("零")
                        append(cardinal(mi.toString())).append("分")
                    }
                    if (sec.isNotEmpty()) append(cardinal(sec)).append("秒")
                }
            }
            t = PERCENT.replace(t) { m ->
                (if (m.groupValues[1].isNotEmpty()) "负" else "") + "百分之" + decimal(m.groupValues[2], m.groupValues[3].ifEmpty { null })
            }
            t = PERMILLE.replace(t) { m ->
                (if (m.groupValues[1].isNotEmpty()) "负" else "") + "千分之" + decimal(m.groupValues[2], m.groupValues[3].ifEmpty { null })
            }
            t = TEMP.replace(t) { m -> m.groupValues[1] + "摄氏度" }
            t = CURRENCY_PRE.replace(t) { m ->
                val unit = CURRENCY_WORDS.getValue(m.groupValues[1])
                val after = m.range.last + 1
                // "¥12元" already carries its unit word
                val hasUnit = unit == "元" && after < t.length && (t[after] == '元' || t[after] == '块')
                m.groupValues[2] + if (hasUnit) "" else unit
            }
            t = UNIT.replace(t) { m -> m.groupValues[1] + UNIT_WORDS.getValue(m.groupValues[2]) }
            t = FRACTION.replace(t) { m -> cardinal(m.groupValues[2]) + "分之" + cardinal(m.groupValues[1]) }
            t = RANGE.replace(t) { m -> m.groupValues[1] + "到" }
            t = NEGATIVE.replace(t) { m -> m.groupValues[1] + "负" }
            t = NUMBER.replace(t) { m ->
                val intPart = m.groupValues[1]
                val frac = m.groupValues[2].ifEmpty { null }
                val end = m.range.last + 1
                val next = if (end < t.length) t[end] else ' '
                if (frac == null && intPart == "2" && next in MEASURE) "两" else numberWord(intPart, frac)
            }
            // hyphen guard from the official zh pipeline
            t = CJK_HYPHEN_CJK.replace(t, "$1，")
            t = OTHER_HYPHEN.replace(t, "$1 ")
            return restoreForTn(t, protected)
        }
    }

    // =====================================================================================
    // English semantic TN
    // =====================================================================================
    internal object EnTn {
        private val ONES = arrayOf(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        )
        private val TENS = arrayOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
        private val ORD = mapOf(
            "one" to "first", "two" to "second", "three" to "third", "five" to "fifth", "eight" to "eighth",
            "nine" to "ninth", "twelve" to "twelfth",
        )

        private fun under1000(n: Int): String {
            val sb = StringBuilder()
            val h = n / 100
            val r = n % 100
            if (h > 0) sb.append(ONES[h]).append(" hundred")
            if (r > 0) {
                if (sb.isNotEmpty()) sb.append(' ')
                if (r < 20) sb.append(ONES[r]) else {
                    sb.append(TENS[r / 10])
                    if (r % 10 > 0) sb.append('-').append(ONES[r % 10])
                }
            }
            return sb.toString()
        }

        fun cardinal(numStr: String): String {
            val s = numStr.trimStart('0')
            if (s.isEmpty()) return "zero"
            if (s.length > 15) return numStr.map { ONES[it - '0'] }.joinToString(" ")
            var n = s.toLong()
            val scales = arrayOf("", " thousand", " million", " billion", " trillion")
            val parts = ArrayList<String>()
            var i = 0
            while (n > 0) {
                val chunk = (n % 1000).toInt()
                if (chunk > 0) parts.add(0, under1000(chunk) + scales[i])
                n /= 1000
                i++
            }
            return parts.joinToString(" ")
        }

        private fun year(s: String): String {
            val y = s.toInt()
            return when {
                y in 2000..2009 -> if (y == 2000) "two thousand" else "two thousand " + ONES[y - 2000]
                y % 100 == 0 -> under1000(y / 100) + " hundred"
                else -> under1000(y / 100) + " " + (if (y % 100 < 10) "oh " + ONES[y % 100] else under1000(y % 100))
            }
        }

        private fun ordinal(numStr: String): String {
            val words = cardinal(numStr)
            val lastSep = maxOf(words.lastIndexOf(' '), words.lastIndexOf('-'))
            val head = words.substring(0, lastSep + 1)
            val last = words.substring(lastSep + 1)
            val ord = ORD[last] ?: if (last.endsWith("y")) last.dropLast(1) + "ieth" else last + "th"
            return head + ord
        }

        private val THOUSANDS = Regex("(?<![\\d.])(\\d{1,3}(?:,\\d{3})+)(?![\\d,])")
        private val CURRENCY = Regex("([$€£])\\s*(\\d+)(?:\\.(\\d{1,2}))?")
        private val PERCENT = Regex("(\\d+(?:\\.\\d+)?)\\s*%")
        private val ORDINAL = Regex("(?<![\\d.])(\\d+)(st|nd|rd|th)\\b")
        private val TIME = Regex("(?<!\\d)([01]?\\d|2[0-3]):([0-5]\\d)(?!\\d)")
        private val YEAR = Regex("(?<![\\d.,])(1[1-9]\\d\\d|20\\d\\d)(?![\\d.,]|\\s*%)")
        private val NUMBER = Regex("(-)?(\\d+)(?:\\.(\\d+))?")
        private val CURRENCY_WORDS = mapOf("$" to "dollars", "€" to "euros", "£" to "pounds")

        fun normalize(input: String): String {
            val protected = ArrayList<String>()
            var t = protectForTn(input, protected)
            t = THOUSANDS.replace(t) { it.value.replace(",", "") }
            t = CURRENCY.replace(t) { m ->
                val unit = CURRENCY_WORDS.getValue(m.groupValues[1])
                val cents = m.groupValues[3]
                cardinal(m.groupValues[2]) + " " + unit + if (cents.isNotEmpty() && cents.toInt() > 0) " " + cardinal(cents) + " cents" else ""
            }
            t = PERCENT.replace(t) { m -> m.groupValues[1] + " percent" }
            t = ORDINAL.replace(t) { m -> ordinal(m.groupValues[1]) }
            t = TIME.replace(t) { m ->
                val mi = m.groupValues[2].toInt()
                cardinal(m.groupValues[1]) + when {
                    mi == 0 -> " o'clock"
                    mi < 10 -> " oh " + ONES[mi]
                    else -> " " + cardinal(mi.toString())
                }
            }
            t = YEAR.replace(t) { m -> year(m.groupValues[1]) }
            t = NUMBER.replace(t) { m ->
                val neg = if (m.groupValues[1].isNotEmpty()) "minus " else ""
                val frac = m.groupValues[3]
                neg + cardinal(m.groupValues[2]) + if (frac.isNotEmpty()) " point " + frac.map { ONES[it - '0'] }.joinToString(" ") else ""
            }
            return restoreForTn(t, protected)
        }
    }
}
