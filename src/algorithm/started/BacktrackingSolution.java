package algorithm.started;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * 回溯算法相关题目
 */
public class BacktrackingSolution {

    /**
     * 21
     *
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     * 示例 1：
     * 输入：l1 = [1,2,4], l2 = [1,3,4]
     * 输出：[1,1,2,3,4,4]
     * 示例 2：
     * 输入：l1 = [], l2 = []
     * 输出：[]
     * 示例 3：
     * 输入：l1 = [], l2 = [0]
     * 输出：[0]
     *
     * 提示：
     * 两个链表的节点数目范围是 [0, 50]
     * -100 <= Node.val <= 100
     * l1 和 l2 均按 非递减顺序 排列
     */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode root = new ListNode();
        ListNode preNode = root;
        while (list1 != null && list2 != null) {
            ListNode node = new ListNode();
            if (list1.val <= list2.val) {
                node.val = list1.val;
                list1 = list1.next;
            } else {
                node.val = list2.val;
                list2 = list2.next;
            }
            preNode.next = node;
            preNode = node;
        }

        if (list1 == null) {
            preNode.next = list2;
        } else {
            preNode.next = list1;
        }

        return root.next;
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
     * 206 反转链表
     *
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
     *
     * 示例 1：
     * 输入：head = [1,2,3,4,5]
     * 输出：[5,4,3,2,1]
     * 示例 2：
     * 输入：head = [1,2]
     * 输出：[2,1]
     * 示例 3：
     * 输入：head = []
     * 输出：[]
     *
     * 提示：
     * 链表中节点的数目范围是 [0, 5000]
     * -5000 <= Node.val <= 5000
     */
    public ListNode reverseList(ListNode head) {
        ListNode preNode = new ListNode(), currNode = new ListNode();
        while (head != null) {
            currNode.val = head.val;
            preNode.next = currNode;
            currNode = preNode;
            preNode = new ListNode();
            head = head.next;
        }

        return currNode.next;
    }

    /**
     * 77 组合
     * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
     * 你可以按 任何顺序 返回答案。
     *
     * 示例 1：
     * 输入：n = 4, k = 2
     * 输出：
     * [
     * [2,4],
     * [3,4],
     * [2,3],
     * [1,2],
     * [1,3],
     * [1,4],
     * ]
     * 示例 2：
     * 输入：n = 1, k = 1
     * 输出：[[1]]
     *
     * 提示：
     * 1 <= n <= 20
     * 1 <= k <= n
     */
    public static List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n < k) {
            return res;
        }
        Deque<Integer> path = new ArrayDeque<>();
//        dfs(n, k, 1, path, res);
        dfs2(1, n, k, path, res);
        return res;
    }

    private static void dfs(int n, int k, int begin, Deque<Integer> path, List<List<Integer>> res) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = begin; i <= n - (k - path.size()) + 1; i++) {
            path.addLast(i);
            System.out.println("递归之前 => " + path);
            dfs(n, k, i + 1, path, res);
            // 重点理解这里：深度优先遍历有回头的过程，因此递归之前做了什么，递归之后需要做相同操作的逆向操作
            path.removeLast();
            System.out.println("递归之后 => " + path);
        }
    }

    private static void dfs2(int begin, int n, int k, Deque<Integer> path, List<List<Integer>> res) {
        if (k == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        // 基础版本的递归终止条件：if (begin == n + 1) {
        if (begin > n - k + 1) {
            return;
        }
        System.out.println("递归之前 => " + path);
        // 不选当前考虑的数 begin，直接递归到下一层
        dfs2(begin + 1, n, k, path, res);


        // 不选当前考虑的数 begin，递归到下一层的时候 k - 1，这里 k 表示还需要选多少个数
        path.addLast(begin);
        System.out.println("递归之中 => " + path);
        dfs2(begin + 1, n, k - 1, path, res);
        // 深度优先遍历有回头的过程，因此需要撤销选择
        path.removeLast();

        System.out.println("递归之后 => " + path);
    }

    /**
     * 46 全排列
     *
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     *
     * 示例 1：
     * 输入：nums = [1,2,3]
     * 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
     * 示例 2：
     * 输入：nums = [0,1]
     * 输出：[[0,1],[1,0]]
     * 示例 3：
     * 输入：nums = [1]
     * 输出：[[1]]
     *
     * 提示：
     * 1 <= nums.length <= 6
     * -10 <= nums[i] <= 10
     * nums 中的所有整数 互不相同
     */
    public static List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        Deque<Integer> path = new LinkedList<>();

        dfs3(nums, path, stack, res);

        return res;
    }

    private static void dfs3(int[] nums, Deque<Integer> path, Stack<Integer> stack, List<List<Integer>> res) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (stack.contains(i)) {
                continue;
            }
            path.add(nums[i]);
            stack.push(i);
            dfs3(nums, path, stack, res);
            stack.pop();
            path.removeLast();
        }
    }

    public List<List<Integer>> permute2(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();

        List<Integer> output = new ArrayList<>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }

    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }

    /**
     * 784. 字母大小写全排列
     *
     * 给定一个字符串s，通过将字符串s中的每个字母转变大小写，我们可以获得一个新的字符串。
     * 返回 所有可能得到的字符串集合 。以 任意顺序 返回输出。
     *
     * 示例 1：
     * 输入：s = "a1b2"
     * 输出：["a1b2", "a1B2", "A1b2", "A1B2"]
     * 示例 2:
     * 输入: s = "3z4"
     * 输出: ["3z4","3Z4"]
     *
     * 提示:
     * 1 <= s.length <= 12
     * s由小写英文字母、大写英文字母和数字组成
     */
    public List<String> letterCasePermutation(String s) {
        List<StringBuilder> ans = new ArrayList();
        ans.add(new StringBuilder());

        for (char c : s.toCharArray()) {
            int n = ans.size();
            if (Character.isLetter(c)) {
                for (int i = 0; i < n; ++i) {
                    ans.add(new StringBuilder(ans.get(i)));
                    ans.get(i).append(Character.toLowerCase(c));
                    ans.get(n + i).append(Character.toUpperCase(c));
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    ans.get(i).append(c);
                }
            }
        }

        List<String> finalans = new ArrayList();
        for (StringBuilder sb : ans) {
            finalans.add(sb.toString());
        }
        return finalans;
    }

    public static void main(String[] args) {
        int[] nums = new int[] {1, 2, 3};
        List<List<Integer>> res = permute(nums);
        System.out.println(res);
    }

}