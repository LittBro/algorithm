package algorithm.started;

/**
 * 二分查找算法题联系
 */
public class BinarySearchSolution {

    /**
     * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
     * <p>
     * 示例 1:
     * 输入: nums = [-1,0,3,5,9,12], target = 9
     * 输出: 4
     * 解释: 9 出现在 nums 中并且下标为 4
     * 示例 2:
     * 输入: nums = [-1,0,3,5,9,12], target = 2
     * 输出: -1
     * 解释: 2 不存在 nums 中因此返回 -1
     * <p>
     * 提示：
     * 你可以假设 nums 中的所有元素是不重复的。
     * n 将在 [1, 10000]之间。
     * nums 的每个元素都将在 [-9999, 9999]之间。
     */
    public int search(int[] nums, int target) {
        int length = nums.length;
        if (nums[0] == target) {
            return 0;
        }
        if (nums[length - 1] == target) {
            return length - 1;
        }
        return twoHalf(0, length, nums, target);
    }

    private int twoHalf(int start, int end, int[] nums, int target) {
        int index = (start + end) / 2;
        if (index == end || index == start) {
            return -1;
        }
        if (nums[index] == target) {
            return index;
        } else if (nums[index] > target) {
            end = index;
        } else if (nums[index] < target) {
            start = index;
        }
        return twoHalf(start, end, nums, target);
    }

    /**
     * 你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。
     * 由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。
     * 假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
     * 你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。
     * 实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。
     * <p>
     * 示例 1：
     * 输入：n = 5, bad = 4
     * 输出：4
     * 解释：
     * 调用 isBadVersion(3) -> false
     * 调用 isBadVersion(5) -> true
     * 调用 isBadVersion(4) -> true
     * 所以，4 是第一个错误的版本。
     * <p>
     * 示例 2：
     * 输入：n = 1, bad = 1
     * 输出：1
     * 提示：
     * 1 <= bad <= n <= 231 - 1
     */
    public int firstBadVersion(int n) {
//        if (n == 1 || isBadVersion(1)) {
//            return 1;
//        }
//        int start = 1;
//        int end = n;
//        while (end > start + 1) {
//            int index = (start + end) / 2;
//            if (isBadVersion(index)) {
//                end = index;
//            } else {
//                start = index;
//            }
//        }
//
//        return end;

        int left = 1, right = n;
        while (left < right) { // 循环直至区间左右端点相同
            int mid = left + (right - left) / 2; // 防止计算时溢出
            if (isBadVersion(mid)) {
                right = mid; // 答案在区间 [left, mid] 中
            } else {
                left = mid + 1; // 答案在区间 [mid+1, right] 中
            }
        }
        // 此时有 left == right，区间缩为一个点，即为答案
        return left;
    }

    private boolean isBadVersion(int version) {
        if (version > 2) {
            return false;
        }
        return true;
    }

    /**
     * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     * <p>
     * 请必须使用时间复杂度为 O(log n) 的算法。
     * <p>
     * 示例 1:
     * 输入: nums = [1,3,5,6], target = 5
     * 输出: 2
     * 示例 2:
     * 输入: nums = [1,3,5,6], target = 2
     * 输出: 1
     * 示例 3:
     * 输入: nums = [1,3,5,6], target = 7
     * 输出: 4
     */
    public int searchInsert(int[] nums, int target) {
        int right = nums.length - 1;
        int left = 0;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[right] < target ? left + 1 : left;
    }

    /**
     * 34
     * 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。
     * 请你找出给定目标值在数组中的开始位置和结束位置。
     * 如果数组中不存在目标值 target，返回[-1, -1]。
     * 你必须设计并实现时间复杂度为O(log n)的算法解决此问题。
     *
     * 示例 1：
     * 输入：nums = [5,7,7,8,8,10], target = 8
     * 输出：[3,4]
     * 示例2：
     * 输入：nums = [5,7,7,8,8,10], target = 6
     * 输出：[-1,-1]
     * 示例 3：
     * 输入：nums = [], target = 0
     * 输出：[-1,-1]
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums.length == 0) {
            return new int[] {-1, -1};
        }
        int start = 0, end = nums.length - 1, mid;

        while (start <= end) {
            mid = (start + end) / 2;
            if (nums[mid] > target) {
                end = mid - 1;
            } else if (nums[mid] < target) {
                start = mid + 1;
            } else {
                while (start < mid && nums[start] != target) {
                    ++start;
                }
                while (end > mid && nums[end] != target) {
                    --end;
                }
                return new int[] {start, end};
            }
        }

        return new int[] {-1, -1};
    }

    /**
     * 33
     * 整数数组 nums 按升序排列，数组中的值 互不相同 。
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，
     * 使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
     * 例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为[4,5,6,7,0,1,2] 。
     * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回-1。
     *
     * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
     *
     * 示例 1：
     * 输入：nums = [4,5,6,7,0,1,2], target = 0
     * 输出：4
     * 示例2：
     * 输入：nums = [4,5,6,7,0,1,2], target = 3
     * 输出：-1
     * 示例 3：
     * 输入：nums = [1], target = 0
     * 输出：-1
     *
     * 提示：
     * 1 <= nums.length <= 5000
     * -104 <= nums[i] <= 104
     * nums 中的每个值都 独一无二
     * 题目数据保证 nums 在预先未知的某个下标上进行了旋转
     * -104 <= target <= 104
     */
    public int search3(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * 74
     * 编写一个高效的算法来判断m x n矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     * 每行中的整数从左到右按升序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     *
     * 示例 1：
     * 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
     * 输出：true
     * 示例 2：
     * 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
     * 输出：false
     *
     * 提示：
     * m == matrix.length
     * n == matrix[i].length
     * 1 <= m, n <= 100
     * -104 <= matrix[i][j], target <= 104
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int start = 0, end = m - 1, mid;
        while (start <= end) {
            mid = (start + end) / 2;
            if (matrix[mid][0] > target) {
                end = mid - 1;
            } else if (matrix[mid][n - 1] < target) {
                start = mid + 1;
            } else {
                int left = 0, right = n - 1, index;
                while (left <= right) {
                    index = (left + right) / 2;
                    if (matrix[mid][index] > target) {
                        right = index - 1;
                    } else if (matrix[mid][index] < target) {
                        left = index + 1;
                    } else {
                        return true;
                    }
                }
                return false;
            }
        }
        return false;
    }

    /**
     * 将二维数组映射为一维数组，然后直接取除数和余数
     */
    public boolean searchMatrix2(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int low = 0, high = m * n - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int x = matrix[mid / n][mid % n];
            if (x < target) {
                low = mid + 1;
            } else if (x > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }

    /**
     * 153
     * 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
     * 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
     * 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
     * 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
     * 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
     * 你必须设计一个时间复杂度为O(log n) 的算法解决此问题。
     *
     * 示例 1：
     * 输入：nums = [3,4,5,1,2]
     * 输出：1
     * 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
     * 示例 2：
     * 输入：nums = [4,5,6,7,0,1,2]
     * 输出：0
     * 解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。
     * 示例 3：
     * 输入：nums = [11,13,15,17]
     * 输出：11
     * 解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
     *
     * 提示：
     * n == nums.length
     * 1 <= n <= 5000
     * -5000 <= nums[i] <= 5000
     * nums 中的所有整数 互不相同
     * nums 原来是一个升序排序的数组，并进行了 1 至 n 次旋转
     */
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1, mid = 0;
        while (left < right) {
            mid = (left + right) / 2;
            if (nums[mid] < nums[left]) {
                if (nums[mid] < nums[mid - 1]) {
                    return nums[mid];
                }
                right = mid;
            } else if (nums[mid] > nums[right]) {
                if (nums[mid] > nums[mid + 1]) {
                    return nums[mid + 1];
                }
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return nums[left];
    }

    public int findMin2(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (nums[pivot] < nums[high]) {
                high = pivot;
            } else {
                low = pivot + 1;
            }
        }
        return nums[low];
    }

    /**
     * 162
     * 峰值元素是指其值严格大于左右相邻值的元素。
     * 给你一个整数数组nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
     * 你可以假设nums[-1] = nums[n] = -∞ 。
     * 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
     *
     * 示例 1：
     * 输入：nums = [1,2,3,1]
     * 输出：2
     * 解释：3 是峰值元素，你的函数应该返回其索引 2。
     * 示例2：
     * 输入：nums = [1,2,1,3,5,6,4]
     * 输出：1 或 5
     * 解释：你的函数可以返回索引 1，其峰值元素为 2；
     * 或者返回索引 5， 其峰值元素为 6。
     *
     * 提示：
     * 1 <= nums.length <= 1000
     * -231 <= nums[i] <= 231 - 1
     * 对于所有有效的 i 都有 nums[i] != nums[i + 1]
     */
    public int findPeakElement(int[] nums) {
        if (nums.length == 1) {
            return 0;
        }

        return loopFind(0, nums.length - 1, nums);
    }

    private int loopFind(int begin, int end, int[] nums) {
        int mid = begin + (end - begin) / 2;

        int result = comparePeed(mid, nums);
        if (result == 2) {
            mid = loopFind(mid + 1, end, nums);
        } else if (result == 0) {
            mid = loopFind(begin, mid - 1, nums);
        }

        return mid;
    }

    private int comparePeed(int mid, int[] nums) {
        if (mid == nums.length - 1) {
            return nums[mid] > nums[mid - 1] ? 1 : 0;
        }

        if (mid == 0) {
            return nums[mid] > nums[mid + 1] ? 1 : 2;
        }

        if (nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) {
            return 1;
        }

        return nums[mid] > nums[mid - 1] ? 2 : 0;
    }
}
