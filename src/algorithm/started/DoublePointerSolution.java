package algorithm.started;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collectors;

/**
 * 双指针算法题
 */
public class DoublePointerSolution {

    /**
     * 977
     * 
     * 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
     * 
     * 示例 1：
     * 输入：nums = [-4,-1,0,3,10]
     * 输出：[0,1,9,16,100]
     * 解释：平方后，数组变为 [16,1,0,9,100]
     * 排序后，数组变为 [0,1,9,16,100]
     * 示例 2：
     * 输入：nums = [-7,-3,2,3,11]
     * 输出：[4,9,9,49,121]
     * 
     * 进阶：
     * 请你设计时间复杂度为 O(n) 的算法解决本问题
     */
    public int[] sortedSquares(int[] nums) {
        int length = nums.length;
        int[] result = new int[length];

        // 定义两个指针
        int left = 0, right = length - 1, index = length - 1;

        while (left <= right) {
            if (Math.abs(nums[left]) < Math.abs(nums[right])) {
                result[index] = nums[right] * nums[right];
                index--;
                right--;
            } else {
                result[index] = nums[left] * nums[left];
                index--;
                left++;
            }
        }

        return result;
    }

    /**
     * 189
     * 
     * 给你一个数组，将数组中的元素向右轮转 k  个位置，其中  k  是非负数。
     * 
     * 示例 1:
     * 输入: nums = [1,2,3,4,5,6,7], k = 3
     * 输出: [5,6,7,1,2,3,4]
     * 解释:
     * 向右轮转 1 步: [7,1,2,3,4,5,6]
     * 向右轮转 2 步: [6,7,1,2,3,4,5]
     * 向右轮转 3 步: [5,6,7,1,2,3,4]
     * 
     * 示例  2:
     * 输入：nums = [-1,-100,3,99], k = 2
     * 输出：[3,99,-1,-100]
     * 解释:
     * 向右轮转 1 步: [99,-1,-100,3]
     * 向右轮转 2 步: [3,99,-1,-100]
     */
    public static void rotate(int[] nums, int k) {
        int length = nums.length;
        int begin = 0;
        int count = 1;
        int preIndex = 0;
        int preValue = nums[preIndex];
        while (count <= nums.length) {
            //将移动后的k值存储起来
            int nowIndex = (preIndex + k) % length;
            int nowValue = nums[nowIndex];
            nums[nowIndex] = preValue;

            count++;

            if (count <= nums.length && nowIndex == begin) {
                begin++;
                nowIndex++;
                nowValue = nums[nowIndex];
            }

            //循环替换
            preIndex = nowIndex;
            preValue = nowValue;

        }
    }

    /**
     * 数组反转，写的太漂亮了
     */
    public void rotate2(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start += 1;
            end -= 1;
        }
    }

    /**
     * 283
     * 
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     * 请注意  ，必须在不复制数组的情况下原地对数组进行操作。
     * 
     * 示例 1:
     * 输入: nums = [0,1,0,3,12]
     * 输出: [1,3,12,0,0]
     * 示例 2:
     * 输入: nums = [0]
     * 输出: [0]
     */
    public static void moveZeroes(int[] nums) {
        int n = nums.length, left = 0, right = 0;
        while (right < n) {
            if (nums[right] != 0) {
                swap(nums, left, right);
                left++;
            }
            right++;
        }
    }

    public static void swap(int[] nums, int left, int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }

    /**
     * 167
     * 
     * 给你一个下标从 1 开始的整数数组  numbers ，该数组已按 非递减顺序排列   ，请你从数组中找出满足相加之和等于目标数  target 的两个数。
     * 如果设这两个数分别是 numbers[index1] 和 numbers[index2] ，则 1 <= index1 < index2 <= numbers.length 。
     * 以长度为 2 的整数数组 [index1, index2] 的形式返回这两个整数的下标 index1 和 index2。
     * 你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。
     * 你所设计的解决方案必须只使用常量级的额外空间。
     * 
     * 
     * 示例 1：
     * 输入：numbers = [2,7,11,15], target = 9
     * 输出：[1,2]
     * 解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。返回 [1, 2] 。
     * 
     * 示例 2：
     * 输入：numbers = [2,3,4], target = 6
     * 输出：[1,3]
     * 解释：2 与 4 之和等于目标数 6 。因此 index1 = 1, index2 = 3 。返回 [1, 3] 。
     * 
     * 示例 3：
     * 输入：numbers = [-1,0], target = -1
     * 输出：[1,2]
     * 解释：-1 与 0 之和等于目标数 -1 。因此 index1 = 1, index2 = 2 。返回 [1, 2] 。
     */
    public int[] twoSum(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            if ((numbers[left] + numbers[right]) > target) {
                right--;
            } else if ((numbers[left] + numbers[right]) < target) {
                left++;
            } else {
                return new int[] {left + 1, right + 1};
            }
        }
        return null;
    }

    /**
     * 344
     * 
     * 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
     * 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
     * 
     * 示例 1：
     * 
     * 输入：s = ["h","e","l","l","o"]
     * 输出：["o","l","l","e","h"]
     * 示例 2：
     * 
     * 输入：s = ["H","a","n","n","a","h"]
     * 输出：["h","a","n","n","a","H"]
     */
    public static void reverseString(char[] s) {
        int left = 0, right = s.length - 1;
        while (left < right) {
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }

    /**
     * 557
     * 
     * 给定一个字符串 s ，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
     * 
     * 示例 1：
     * 输入：s = "Let's take LeetCode contest"
     * 输出："s'teL ekat edoCteeL tsetnoc"
     * 示例 2:
     * 输入： s = "God Ding"
     * 输出："doG gniD"
     */
    public static String reverseWords(String s) {
        StringBuffer result = new StringBuffer();
        String[] s1 = s.split(" ");
        for (int i = 0; i < s1.length; i++) {
            char[] word = s1[i].toCharArray();
            reverseString(word);
            for (int j = 0; j < word.length; j++) {
                result.append(word[j]);
            }
            result.append(" ");
        }
        return result.substring(0, result.length() - 1);
    }

    /**
     * 876
     * 
     * 给定一个头结点为 head  j的非空单链表，返回链表的中间结点。
     * 如果有两个中间结点，则返回第二个中间结点。
     * 
     * 示例 1：
     * 输入：[1,2,3,4,5]
     * 输出：此列表中的结点 3 (序列化形式：[3,4,5])
     * 返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
     * 注意，我们返回了一个 ListNode 类型的对象 ans，这样：
     * ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
     * 示例  j2：
     * 输入：[1,2,3,4,5,6]
     * 输出：此列表中的结点 4 (序列化形式：[4,5,6])
     * 由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
     */
    public ListNode middleNode(ListNode head) {
        int count = 1;
        ListNode node = head;
        while (node.next != null) {
            node = node.next;
            count++;
        }
        count = count / 2;
        node = head;
        while (count > 0) {
            node = node.next;
            count--;
        }
        return node;
    }

    /**
     * 使用快慢指针，快指针一次走两步，慢指针一次走一步
     */
    public ListNode middleNode2(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    /**
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     * 
     * 
     * 示例 1：
     * 输入：head = [1,2,3,4,5], n = 2
     * 输出：[1,2,3,5]
     * 
     * 示例 2：
     * 输入：head = [1], n = 1
     * 输出：[]
     * 
     * 示例 3：
     * 输入：head = [1,2], n = 1
     * 输出：[1]
     */

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; ++i) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        ListNode ans = dummy.next;
        return ans;
    }

    /**
     * 82
     * 给定一个已排序的链表的头head ，删除原始链表中所有重复数字的节点，只留下不同的数字。返回 已排序的链表。
     *
     * 示例 1：
     * 输入：head = [1,2,3,3,4,4,5]
     * 输出：[1,2,5]
     * 示例 2：
     * 输入：head = [1,1,1,2,3]
     * 输出：[2,3]
     *
     * 提示：
     * 链表中节点数目在范围 [0, 300] 内
     * -100 <= Node.val <= 100
     * 题目数据保证链表已经按升序 排列
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode();
        ListNode index = new ListNode(101);
        root.next = index;
        ListNode pre = index, curr = head;
        while (curr.next != null) {
            if (curr.val != pre.val && curr.val != curr.next.val) {
                index.next = curr;
                index = curr;
            }
            pre = curr;
            curr = curr.next;
        }

        if (curr.val != pre.val) {
            index.next = curr;
        } else {
            index.next = null;
        }
        return root.next.next;
    }

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null) {
            return head;
        }

        ListNode dummy = new ListNode(0, head);

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int x = cur.next.val;
                while (cur.next != null && cur.next.val == x) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }

        return dummy.next;
    }

    /**
     * 15
     * 给你一个包含 n 个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
     * 注意：答案中不可以包含重复的三元组。
     *
     * 示例 1：
     * 输入：nums = [-1,0,1,2,-1,-4]
     * 输出：[[-1,-1,2],[-1,0,1]]
     * 示例 2：
     * 输入：nums = []
     * 输出：[]
     * 示例 3：
     * 输入：nums = [0]
     * 输出：[]
     *
     * 提示：
     * 0 <= nums.length <= 3000
     * -105 <= nums[i] <= 105
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums.length < 3) {
            return result;
        }

        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 2 && nums[i] <= 0; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) {
                    right--;
                    while (left < right && nums[right] == nums[right + 1]) {
                        right--;
                    }
                } else if (sum < 0) {
                    left++;
                    while (left < right && nums[left] == nums[left - 1]) {
                        left++;
                    }
                } else {
                    List<Integer> list = Arrays.asList(nums[i], nums[left], nums[right]);
                    res.add(list);
                    left++;
                    while (left < right && nums[left] == nums[left - 1]) {
                        left++;
                    }
                }
            }
        }

        return res;
    }

    public List<List<Integer>> threeSum2(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    /**
     * 844
     * 给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。
     * 注意：如果对空文本输入退格字符，文本继续为空。
     *
     * 示例 1：
     * 输入：s = "ab#c", t = "ad#c"
     * 输出：true
     * 解释：s 和 t 都会变成 "ac"。
     * 示例 2：
     * 输入：s = "ab##", t = "c#d#"
     * 输出：true
     * 解释：s 和 t 都会变成 ""。
     * 示例 3：
     * 输入：s = "a#c", t = "b"
     * 输出：false
     * 解释：s 会变成 "c"，但 t 仍然是 "b"。
     *
     * 提示：
     * 1 <= s.length, t.length <= 200
     * s 和 t 只含有小写字母以及字符 '#'
     *
     * 进阶：
     * 你可以用 O(n) 的时间复杂度和 O(1) 的空间复杂度解决该问题吗？
     */
    public boolean backspaceCompare(String s, String t) {
        char[] sChar = s.toCharArray();
        char[] tChar = t.toCharArray();

        Stack<Character> sStack = new Stack<>();
        Stack<Character> tStack = new Stack<>();
        for (int i = 0; i < sChar.length; i++) {
            if (sChar[i] == '#') {
                if (sStack.size() > 0) {
                    sStack.pop();
                }
            } else {
                sStack.push(sChar[i]);
            }
        }
        for (int i = 0; i < tChar.length; i++) {
            if (tChar[i] == '#') {
                if (tStack.size() > 0) {
                    tStack.pop();
                }
            } else {
                tStack.push(tChar[i]);
            }
        }
        return sStack.equals(tStack);
    }

    /**
     * 986
     * 给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而secondList[j] = [startj, endj] 。
     * 每个区间列表都是成对 不相交 的，并且 已经排序 。
     * 返回这 两个区间列表的交集 。
     * 形式上，闭区间[a, b]（其中a <= b）表示实数x的集合，而a <= x <= b 。
     * 两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。
     *
     * 示例 1：
     * 输入：firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
     * 输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
     * 示例 2：
     * 输入：firstList = [[1,3],[5,9]], secondList = []
     * 输出：[]
     * 示例 3：
     * 输入：firstList = [], secondList = [[4,8],[10,12]]
     * 输出：[]
     * 示例 4：
     * 输入：firstList = [[1,7]], secondList = [[3,10]]
     * 输出：[[3,7]]
     */
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        int a = 0, b = 0;
        List<int[]> result = new ArrayList<>();
        while (a < firstList.length && b < secondList.length) {
            while (firstList[a][1] < secondList[b][0]) {
                a++;
                if (a == firstList.length) {
                    return result.toArray(new int[0][]);
                }
            }
            while (firstList[a][0] > secondList[b][1]) {
                b++;
                if (b == secondList.length) {
                    return result.toArray(new int[0][]);
                }
            }
            if (firstList[a][1] < secondList[b][0]) {
                a++;
                continue;
            }

            int[] array = new int[] {Math.max(firstList[a][0], secondList[b][0]), Math.min(firstList[a][1], secondList[b][1])};
            result.add(array);
            if (firstList[a][1] < secondList[b][1]) {
                a++;
            } else {
                b++;
            }
        }

        return result.toArray(new int[0][]);
    }

    public int[][] intervalIntersection2(int[][] A, int[][] B) {
        List<int[]> ans = new ArrayList();
        int i = 0, j = 0;

        while (i < A.length && j < B.length) {
            // Let's check if A[i] intersects B[j].
            // lo - the startpoint of the intersection
            // hi - the endpoint of the intersection
            int lo = Math.max(A[i][0], B[j][0]);
            int hi = Math.min(A[i][1], B[j][1]);
            if (lo <= hi) {
                ans.add(new int[] {lo, hi});
            }

            // Remove the interval with the smallest endpoint
            if (A[i][1] < B[j][1]) {
                i++;
            } else {
                j++;
            }
        }

        return ans.toArray(new int[ans.size()][]);
    }

    /**
     * 11
     * 给定一个长度为 n 的整数数组height。有n条垂线，第 i 条线的两个端点是(i, 0)和(i, height[i])。
     * 找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。
     * 说明：你不能倾斜容器。
     *
     * 示例 1：
     * 输入：[1,8,6,2,5,4,8,3,7]
     * 输出：49
     * 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为49。
     * 示例 2：
     * 输入：height = [1,1]
     * 输出：1
     *
     * 提示：
     * n == height.length
     * 2 <= n <= 105
     * 0 <= height[i] <= 104
     */
    public int maxArea(int[] height) {
        int maxLeft = 0;
        int result = 0;
        for (int i = 0; i < height.length - 1; i++) {
            if (i > 0 && height[i] < maxLeft) {
                continue;
            } else {
                maxLeft = height[i];
            }

            int maxRight = 0;
            for (int j = height.length - 1; j > i; j--) {
                if (j < height.length - 1 && height[j] < maxRight) {
                    continue;
                } else {
                    maxRight = height[j];
                }
                result = Math.max(result, (j - i) * Math.min(height[i], height[j]));
            }
        }
        return result;
    }

    public int maxArea2(int[] height) {
        int left = 0, right = height.length - 1;
        int result = 0;
        while (left < right) {
            result = Math.max(result, (right - left) * Math.min(height[right], height[left]));
            if (height[right] > height[left]) {
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    /**
     * 61 旋转链表
     *
     * 给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
     * 输入：head = [1,2,3,4,5], k = 2
     * 输出：[4,5,1,2,3]
     */

    public static void main(String[] args) {
        Set<List<Integer>> set = new HashSet<>();
        List<Integer> list = Arrays.asList(0, 1, -1);
        if (list.stream().reduce(Integer::sum).get() == 0) {
            List<Integer> sortList = list.stream().sorted().collect(Collectors.toList());
            set.add(sortList);
        }
        System.out.println(list);
        System.out.println(set);
    }

}
