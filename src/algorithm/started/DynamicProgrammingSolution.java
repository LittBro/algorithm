package algorithm.started;

import java.util.List;

public class DynamicProgrammingSolution {

    /**
     * 假设你正在爬楼梯。需要 n阶你才能到达楼顶。
     *
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     *
     * 示例 1：
     * 输入：n = 2
     * 输出：2
     * 解释：有两种方法可以爬到楼顶。
     * 1. 1 阶 + 1 阶
     * 2. 2 阶
     * 示例 2：
     * 输入：n = 3
     * 输出：3
     * 解释：有三种方法可以爬到楼顶。
     * 1. 1 阶 + 1 阶 + 1 阶
     * 2. 1 阶 + 2 阶
     * 3. 2 阶 + 1 阶
     *
     * 提示：
     * 1 <= n <= 45
     */
    public int climbStairs(int n) {
        int res = 1;
        int p, q = 1;
        for (int i = 1; i < n; i++) {
            p = q;
            q = res;
            res = p + q;
        }

        return res;
    }


    /**
     * 198 打家劫舍
     *
     * 你是一个专业的小偷，计划偷窃沿街的房屋。
     * 每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
     * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     *
     * 示例 1：
     * 输入：[1,2,3,1]
     * 输出：4
     * 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     * 偷窃到的最高金额 = 1 + 3 = 4 。
     *
     * 示例 2：
     * 输入：[2,7,9,3,1]
     * 输出：12
     * 解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     * 偷窃到的最高金额 = 2 + 9 + 1 = 12
     *
     * 提示：
     * 1 <= nums.length <= 100
     * 0 <= nums[i] <= 400
     */
    public int rob(int[] nums) {
        int length = nums.length;
        if (length == 1) {
            return nums[0];
        }
        int pre = nums[0];
        int curr = Math.max(nums[0], nums[1]);
        for (int i = 2; i < length; i++) {
            int temp = Math.max(pre + nums[i], curr);
            pre = curr;
            curr = temp;
        }
        return curr;
    }

    /**
     * 120
     * 给定一个三角形 triangle ，找出自顶向下的最小路径和。
     * 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
     * 也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
     *
     * 示例 1：
     * 输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
     * 输出：11
     * 解释：如下面简图所示：
     * 2
     * 3 4
     * 6 5 7
     * 4 1 8 3
     * 自顶向下的最小路径和为11（即，2+3+5+1= 11）。
     * 示例 2：
     * 输入：triangle = [[-10]]
     * 输出：-10
     *
     * 提示：
     * 1 <= triangle.length <= 200
     * triangle[0].length == 1
     * triangle[i].length == triangle[i - 1].length + 1
     * -104 <= triangle[i][j] <= 104
     *
     * 进阶：
     * 你可以只使用 O(n)的额外空间（n 为三角形的总行数）来解决这个问题吗？
     */
    public static int minimumTotal(List<List<Integer>> triangle) {
        int length = triangle.size();
        int[] res = new int[length];
        res[0] = triangle.get(0).get(0);

        for (int i = 1; i < length; ++i) {
            List<Integer> integers = triangle.get(i);
            for (int j = i - 1; j > 0; j--) {
                res[j] = Math.min(res[j - 1], res[j]) + integers.get(j);
            }
            res[0] += integers.get(0);
        }

        int min = res[0];
        for (int i = 0; i < length; i++) {
            min = Math.min(res[i], min);
        }
        return min;
    }
}
