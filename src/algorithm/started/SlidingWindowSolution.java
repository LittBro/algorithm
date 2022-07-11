package algorithm.started;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 滑动窗口算法题
 */
public class SlidingWindowSolution {

    /**
     * 3
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
     * <p>
     * 示例  1:
     * 输入: s = "abcabcbb"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
     * 示例 2:
     * 输入: s = "bbbbb"
     * 输出: 1
     * 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
     * 示例 3:
     * 输入: s = "pwwkew"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是  "wke"，所以其长度为 3。
     * <p>
     * 请注意，你的答案必须是 子串 的长度，"pwke"  是一个子序列，不是子串。
     */
    public static int lengthOfLongestSubstring(String s) {
        if (s.isEmpty()) {
            return 0;
        }
        int left = 0, right = 2, max = 1;
        while (right <= s.length()) {
            if (checkRepeat(s.substring(left, right))) {
                left++;
                right++;
            } else {
                max++;
                right++;
            }
        }

        return max;
    }

    private static boolean checkRepeat(String string) {
        if (string.isBlank()) {
            return true;
        }

        Set<Character> charSet = new HashSet<>();
        char[] chars = string.toCharArray();
        for (char aChar : chars) {
            charSet.add(aChar);
        }

        return charSet.size() != chars.length;
    }

    /**
     * 最长的子串必然是从节点的一个位置开始的，只要遍历字符串，必然能找到这个起始位置
     */
    public int lengthOfLongestSubstring2(String s) {
        // 哈希集合，记录每个字符是否出现过
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.remove(s.charAt(i - 1));
            }
            while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))) {
                // 不断地移动右指针
                occ.add(s.charAt(rk + 1));
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = Math.max(ans, rk - i + 1);
        }
        return ans;
    }

    /**
     * 567
     * 给你两个字符串  s1  和  s2 ，写一个函数来判断 s2 是否包含 s1  的排列。如果是，返回 true ；否则，返回 false 。
     * <p>
     * 换句话说，s1 的排列之一是 s2 的 子串 。
     * <p>
     * 示例 1：
     * 输入：s1 = "ab" s2 = "eidbaooo"
     * 输出：true
     * 解释：s2 包含 s1 的排列之一 ("ba").
     * 示例 2：
     * 输入：s1= "ab" s2 = "eidboaoo"
     * 输出：false
     */
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) {
            return false;
        }
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            char key = s1.charAt(i);
            int value = map.getOrDefault(key, 0);
            map.put(key, ++value);
        }

        String s3 = s2.substring(0, s1.length());
        for (int i = 0; i < s3.length(); i++) {
            char key = s1.charAt(i);
            int value = map.getOrDefault(key, 0);
            --value;
            if (value == 0) {
                map.remove(key);
            } else {
                map.put(key, value);
            }
        }

        int left = 0;
        int right = s1.length();
        while (right < s2.length()) {
            if (map.size() == 0) {
                return true;
            }
            int value1 = map.get(s2.charAt(left));
            map.put(s1.charAt(left), ++value1);
            ++left;

            int value2 = map.getOrDefault(s2.charAt(right), 0);
            --value2;
            if (value2 == 0) {
                map.remove(s2.charAt(right));
            } else {
                map.put(s2.charAt(right), value2);
            }

        }

        return true;
    }

    public boolean checkInclusion2(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }
        // 我一开始考虑的是用Map存储，但其实可以用数组存储，逻辑就会更清晰一些
        // 但这里存在的问题则是用了两个数组去做比较，导致效率降低
        int[] cnt1 = new int[26];
        int[] cnt2 = new int[26];
        for (int i = 0; i < n; ++i) {
            ++cnt1[s1.charAt(i) - 'a'];
            ++cnt2[s2.charAt(i) - 'a'];
        }
        if (Arrays.equals(cnt1, cnt2)) {
            return true;
        }
        for (int i = n; i < m; ++i) {
            ++cnt2[s2.charAt(i) - 'a'];
            --cnt2[s2.charAt(i - n) - 'a'];
            if (Arrays.equals(cnt1, cnt2)) {
                return true;
            }
        }
        return false;
    }

    public static boolean checkInclusion3(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }

        int[] cnt = new int[26];
        for (int i = 0; i < n; i++) {
            --cnt[s1.charAt(i) - 'a'];
            ++cnt[s2.charAt(i) - 'a'];
        }

        int difference = 0;
        for (int j : cnt) {
            difference += Math.abs(j);
        }
        if (difference == 0) {
            return true;
        }

        for (int i = n; i < m; i++) {
            int x = s2.charAt(i) - 'a', y = s2.charAt(i - n) - 'a';
            if (x == y) {
                continue;
            }
            if (cnt[x] >= 0) {
                difference++;
            } else {
                difference--;
            }
            ++cnt[x];

            if (cnt[y] > 0) {
                difference--;
            } else {
                difference++;
            }
            --cnt[y];
            if (difference == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * 依然是用26个大小的数组去存储s1的情况
     * 但是在比较s2是否包含s1的排列时，采用双指正的形式，因为如果存在子排序序列，那一定个数也是相等的
     */
    public boolean checkInclusion4(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }
        int[] cnt = new int[26];
        for (int i = 0; i < n; ++i) {
            --cnt[s1.charAt(i) - 'a'];
        }
        int left = 0;
        for (int right = 0; right < m; ++right) {
            int x = s2.charAt(right) - 'a';
            ++cnt[x];
            while (cnt[x] > 0) {
                --cnt[s2.charAt(left) - 'a'];
                ++left;
            }
            if (right - left + 1 == n) {
                return true;
            }
        }
        return false;
    }

    /**
     * 438
     * 给定两个字符串s和 p，找到s中所有p的异位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
     * 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
     *
     * 示例1:
     * 输入: s = "cbaebabacd", p = "abc"
     * 输出: [0,6]
     * 解释:
     * 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
     * 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
     * 示例 2:
     * 输入: s = "abab", p = "ab"
     * 输出: [0,1,2]
     * 解释:
     * 起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
     * 起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
     * 起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
     *
     * 提示:
     * 1 <= s.length, p.length <= 3 * 104
     * s和p仅包含小写字母
     */
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        if (p.length() > s.length()) {
            return result;
        }

        int length = p.length(), diff = 0;
        int[] charArray = new int[26];
        for (int i = 0; i < length; i++) {
            char c = p.charAt(i);
            ++charArray[c - 'a'];
            ++diff;
        }

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);

            if (i > length - 1) {
                char l = s.charAt(i - length);
                if (c == l) {
                    if (diff == 0) {
                        result.add(i - length + 1);
                    }
                    continue;
                }
                ++charArray[l - 'a'];
                if (charArray[l - 'a'] <= 0) {
                    --diff;
                } else {
                    ++diff;
                }
            }

            --charArray[c - 'a'];
            if (charArray[c - 'a'] >= 0) {
                --diff;
            } else {
                ++diff;
            }

            if (diff == 0) {
                result.add(i - length + 1);
            }
        }

        return result;
    }

    /**
     * 713
     * 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
     *
     * 示例 1：
     * 输入：nums = [10,5,2,6], k = 100
     * 输出：8
     * 解释：8 个乘积小于 100 的子数组分别为：[10]、[5]、[2],、[6]、[10,5]、[5,2]、[2,6]、[5,2,6]。
     * 需要注意的是 [10,5,2] 并不是乘积小于 100 的子数组。
     * 示例 2：
     * 输入：nums = [1,2,3], k = 0
     * 输出：0
     *
     * 提示:
     * 1 <= nums.length <= 3 * 104
     * 1 <= nums[i] <= 1000
     * 0 <= k <= 106
     */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            int times = 0;
            int p = 1;
            while (times < nums.length - i) {
                p *= nums[i + times];
                ++times;
                if (p < k) {
                    ++res;
                } else {
                    times = nums.length;
                }
            }
        }
        return res;
    }

    public int numSubarrayProductLessThanK2(int[] nums, int k) {
        int n = nums.length, ret = 0;
        int prod = 1, i = 0;
        for (int j = 0; j < n; j++) {
            prod *= nums[j];
            while (i <= j && prod >= k) {
                prod /= nums[i];
                i++;
            }
            ret += j - i + 1;
        }
        return ret;
    }

    /**
     * 209. 长度最小的子数组
     *
     * 给定一个含有n个正整数的数组和一个正整数 target 。
     * 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组[numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
     *
     * 示例 1：
     * 输入：target = 7, nums = [2,3,1,2,4,3]
     * 输出：2
     * 解释：子数组[4,3]是该条件下的长度最小的子数组。
     * 示例 2：
     * 输入：target = 4, nums = [1,4,4]
     * 输出：1
     * 示例 3：
     * 输入：target = 11, nums = [1,1,1,1,1,1,1,1]
     * 输出：0
     *
     * 提示：
     * 1 <= target <= 109
     * 1 <= nums.length <= 105
     * 1 <= nums[i] <= 105
     *
     * 进阶：
     * 如果你已经实现 O(n) 时间复杂度的解法, 请尝试设计一个 O(n log(n)) 时间复杂度的解法。
     */
    public static int minSubArrayLen(int target, int[] nums) {
        int l = 0, r = 0, res = Integer.MAX_VALUE;
        int prod = 0;
        while (r < nums.length) {
            prod += nums[r];
            while (prod >= target) {
                int len = r - l + 1;
                if (len == 1) {
                    return 1;
                } else if (len < res) {
                    res = len;
                }
                prod -= nums[l];
                ++l;
            }
            ++r;

        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    public static int minSubArrayLen2(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = Math.min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    public static void main(String[] args) {
        System.out.println(minSubArrayLen(15, new int[] {1, 2, 3, 4, 5}));
    }
}
