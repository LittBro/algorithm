package algorithm.started;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

/**
 * 广度或者深度优先搜索相关的算法题
 * <p>
 * 广度优先搜索是借助栈进行入栈出栈
 * 深度优先搜索则是使用递归进行查询
 */
public class BFSOrDFSSolution {

    /**
     * 733 图像渲染
     * <p>
     * 有一幅以  m x n  的二维整数数组表示的图画  image  ，其中  image[i][j]  表示该图画的像素值大小。
     * 你也被给予三个整数 sr ,   sc 和 newColor 。你应该从像素  image[sr][sc]  开始对图像进行 上色填充 。
     * 为了完成 上色工作 ，从初始像素开始，记录初始坐标的 上下左右四个方向上 像素值与初始坐标相同的相连像素点，
     * 接着再记录这四个方向上符合条件的像素点与他们对应 四个方向上 像素值与初始坐标相同的相连像素点，……，重复该过程。
     * 将所有有记录的像素点的颜色值改为  newColor  。
     * 最后返回 经过上色渲染后的图像  。
     * <p>
     * 示例 1:
     * 输入: image = [[1,1,1],[1,1,0],[1,0,1]]，sr = 1, sc = 1, newColor = 2
     * 输出: [[2,2,2],[2,2,0],[2,0,1]]
     * 解析: 在图像的正中间，(坐标(sr,sc)=(1,1)),在路径上所有符合条件的像素点的颜色都被更改成2。
     * 注意，右下角的像素没有更改为2，因为它不是在上下左右四个方向上与初始点相连的像素点。
     * <p>
     * 示例 2:
     * 输入: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, newColor = 2
     * 输出: [[2,2,2],[2,2,2]]
     */
    public static int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int color = image[sr][sc];
        if (color == newColor) {
            return image;
        }

        int iWidth = image.length;
        int iLong = image[0].length;
        Stack<Node2> stack = new Stack<>();
        stack.push(new Node2(sr, sc));
        while (!stack.isEmpty()) {
            Node2 Node2 = stack.pop();
            int left = Node2.getLeft(), right = Node2.getRight();
            image[left][right] = newColor;
            if (left - 1 >= 0 && image[left - 1][right] == color) {
                image[left - 1][right] = newColor;
                stack.push(new Node2(left - 1, right));
            }
            if (left + 1 < iWidth && image[left + 1][right] == color) {
                image[left + 1][right] = newColor;
                stack.push(new Node2(left + 1, right));
            }
            if (right - 1 >= 0 && image[left][right - 1] == color) {
                image[left][right - 1] = newColor;
                stack.push(new Node2(left, right - 1));
            }
            if (right + 1 < iLong && image[left][right + 1] == color) {
                image[left][right + 1] = newColor;
                stack.push(new Node2(left, right + 1));
            }
        }

        return image;
    }

    static class Node2 {
        int left;
        int right;

        Node2(int left, int right) {
            this.left = left;
            this.right = right;
        }

        public int getLeft() {
            return left;
        }

        public int getRight() {
            return right;
        }
    }

    /**
     * 深度优先搜素的方案
     */
    int[] dx = {1, 0, 0, -1};
    int[] dy = {0, 1, -1, 0};

    public int[][] floodFill2(int[][] image, int sr, int sc, int newColor) {
        int currColor = image[sr][sc];
        if (currColor != newColor) {
            dfs(image, sr, sc, currColor, newColor);
        }
        return image;
    }

    public void dfs(int[][] image, int x, int y, int color, int newColor) {
        if (image[x][y] == color) {
            image[x][y] = newColor;
            for (int i = 0; i < 4; i++) {
                int mx = x + dx[i], my = y + dy[i];
                if (mx >= 0 && mx < image.length && my >= 0 && my < image[0].length) {
                    dfs(image, mx, my, color, newColor);
                }
            }
        }
    }

    /**
     * 695
     * <p>
     * 给你一个大小为 m x n 的二进制矩阵 grid 。
     * 岛屿是由一些相邻的1(代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。
     * 你可以假设grid 的四个边缘都被 0（代表水）包围着。
     * 岛屿的面积是岛上值为 1 的单元格的数目。
     * 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
     * <p>
     * 示例 1：
     * 输入：grid =
     * [[0,0,1,0,0,0,0,1,0,0,0,0,0]
     * ,[0,0,0,0,0,0,0,1,1,1,0,0,0]
     * ,[0,1,1,0,1,0,0,0,0,0,0,0,0]
     * ,[0,1,0,0,1,1,0,0,1,0,1,0,0]
     * ,[0,1,0,0,1,1,0,0,1,1,1,0,0]
     * ,[0,0,0,0,0,0,0,0,0,0,1,0,0]
     * ,[0,0,0,0,0,0,0,1,1,1,0,0,0]
     * ,[0,0,0,0,0,0,0,1,1,0,0,0,0]]
     * 输出：6
     * 解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
     * 示例 2：
     * 输入：grid = [[0,0,0,0,0,0,0,0]]
     * 输出：0
     */
    public static int maxAreaOfIsland(int[][] grid) {
        int maxLand = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                maxLand = Math.max(dfs(grid, i, j, 0), maxLand);
            }
        }

        return maxLand;
    }

    /**
     * 深度优先搜素的方案
     */
    static int[] dx2 = {1, 0, 0, -1};
    static int[] dy2 = {0, 1, -1, 0};

    private static int dfs(int[][] grid, int sr, int sc, int land) {
        if (sr >= grid.length || sr < 0 || sc >= grid[0].length || sc < 0 || grid[sr][sc] == 0) {
            return land;
        }

        grid[sr][sc] = 0;
        land++;
        for (int i = 0; i < 4; i++) {
            land = dfs(grid, sr + dx2[i], sc + dy2[i], land);
        }

        return land;
    }

    /**
     * 617 合并二叉树
     * <p>
     * 给你两棵二叉树： root1 和 root2 。
     * 想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。
     * 你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。
     * 返回合并后的二叉树。
     * 注意: 合并过程必须从两个树的根节点开始。
     * <p>
     * 示例 1：
     * 输入：root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
     * 输出：[3,4,5,5,4,null,7]
     * <p>
     * 示例 2：
     * 输入：root1 = [1], root2 = [1,2]
     * 输出：[2,2]
     */
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        } else if (root2 == null) {
            return root1;
        }

        frontLoop(root1, root2);

        return root1;
    }

    private void frontLoop(TreeNode node, TreeNode node2) {

        node.val = node.val + node2.val;

        if (node.left != null && node2.left != null) {
            frontLoop(node.left, node2.left);
        }
        if (node.left == null && node2.left != null) {
            node.left = node2.left;
        }

        if (node.right != null && node2.right != null) {
            frontLoop(node.right, node2.right);
        }
        if (node.right == null && node2.right != null) {
            node.right = node2.right;
        }

    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    /**
     * 116
     * <p>
     * 给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
     * struct Node {
     * int val;
     * Node *left;
     * Node *right;
     * Node *next;
     * }
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     * 初始状态下，所有next 指针都被设置为 NULL。
     * <p>
     * 示例 1：
     * 输入：root = [1,2,3,4,5,6,7]
     * 输出：[1,#,2,3,#,4,5,6,7,#]
     * 解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
     * 示例 2:
     * 输入：root = []
     * 输出：[]
     */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }

        // 创建队列用于广度优先搜索
        Queue<Node> currQueue = new LinkedList<>();
        Queue<Node> nextQueue = new LinkedList<>();
        currQueue.add(root);

        Node preNode = null, currNode;
        while (!currQueue.isEmpty()) {
            currNode = currQueue.poll();
            if (preNode != null) {
                preNode.next = currNode;
            }
            if (currNode.left != null) {
                nextQueue.offer(currNode.left);
                nextQueue.offer(currNode.right);
            }
            preNode = currNode;

            if (currQueue.isEmpty() && !nextQueue.isEmpty()) {
                currQueue = nextQueue;
                nextQueue = new LinkedList<>();
                preNode = null;
            }
        }

        return root;
    }

    public Node connect2(Node root) {
        if (root == null) {
            return root;
        }

        // 从根节点开始
        Node leftmost = root;

        while (leftmost.left != null) {

            // 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
            Node head = leftmost;

            while (head != null) {

                // CONNECTION 1
                head.left.next = head.right;

                // CONNECTION 2
                // 这一步的思路挺亮眼的，虽然对于当前层节点来说，找不到它的双胞胎节点，但是通过它的父节点，能找到父节点next节点的子节点。
                if (head.next != null) {
                    head.right.next = head.next.left;
                }

                // 指针向后移动
                head = head.next;
            }

            // 去下一层的最左的节点
            leftmost = leftmost.left;
        }

        return root;
    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    /**
     * 542 01矩阵
     * <p>
     * 给定一个由 0 和 1 组成的矩阵 mat，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
     * 两个相邻元素间的距离为 1 。
     * <p>
     * 示例 1：
     * 输入：mat = [[0,0,0],[0,1,0],[0,0,0]]
     * 输出：[[0,0,0],[0,1,0],[0,0,0]]
     * 示例 2：
     * 输入：mat = [[0,0,0],[0,1,0],[1,1,1]]
     * 输出：[[0,0,0],[0,1,0],[1,2,1]]
     */
    static int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    public int[][] updateMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dist = new int[m][n];
        boolean[][] seen = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<int[]>();
        // 将所有的 0 添加进初始队列中
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    queue.offer(new int[] {i, j});
                    seen[i][j] = true;
                }
            }
        }

        // 广度优先搜索
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int i = cell[0], j = cell[1];
            for (int d = 0; d < 4; ++d) {
                int ni = i + dirs[d][0];
                int nj = j + dirs[d][1];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && !seen[ni][nj]) {
                    dist[ni][nj] = dist[i][j] + 1;
                    queue.offer(new int[] {ni, nj});
                    seen[ni][nj] = true;
                }
            }
        }

        return dist;
    }

    public int[][] updateMatrix2(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 初始化动态规划的数组，所有的距离值都设置为一个很大的数
        int[][] dist = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dist[i], Integer.MAX_VALUE / 2);
        }
        // 如果 (i, j) 的元素为 0，那么距离为 0
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    dist[i][j] = 0;
                }
            }
        }
        // 只有 水平向左移动 和 竖直向上移动，注意动态规划的计算顺序
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i - 1 >= 0) {
                    dist[i][j] = Math.min(dist[i][j], dist[i - 1][j] + 1);
                }
                if (j - 1 >= 0) {
                    dist[i][j] = Math.min(dist[i][j], dist[i][j - 1] + 1);
                }
            }
        }
        // 只有 水平向右移动 和 竖直向下移动，注意动态规划的计算顺序
        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                if (i + 1 < m) {
                    dist[i][j] = Math.min(dist[i][j], dist[i + 1][j] + 1);
                }
                if (j + 1 < n) {
                    dist[i][j] = Math.min(dist[i][j], dist[i][j + 1] + 1);
                }
            }
        }
        return dist;
    }

    /**
     * 994
     * <p>
     * 在给定的m x n网格grid中，每个单元格可以有以下三个值之一：
     * 值0代表空单元格；
     * 值1代表新鲜橘子；
     * 值2代表腐烂的橘子。
     * 每分钟，腐烂的橘子周围4 个方向上相邻 的新鲜橘子都会腐烂。
     * 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回-1。
     * <p>
     * 示例 1：
     * 输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
     * 输出：4
     * 示例 2：
     * 输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
     * 输出：-1
     * 解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个正向上。
     * 示例 3：
     * 输入：grid = [[0,2]]
     * 输出：0
     * 解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
     */
    public int orangesRotting(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int fineCount = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[] {i, j});
                    grid[i][j] = 0;
                }
                if (grid[i][j] == 1) {
                    ++fineCount;
                }
            }
        }

        int count = queue.size();
        int res = 0;
        while (count > 0) {
            --count;
            int[] pos = queue.poll();
            for (int i = 0; i < 4; i++) {
                int x = pos[0] + dirs[i][0];
                int y = pos[1] + dirs[i][1];
                if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 1) {
                    queue.offer(new int[] {x, y});
                    grid[x][y] = 0;
                    --fineCount;
                }
            }
            if (count == 0 && queue.size() != 0) {
                ++res;
                count = queue.size();
            }
        }

        return fineCount == 0 ? res : -1;
    }

    /**
     * 200. 岛屿数量
     *
     * 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     *
     * 示例 1：
     * 输入：grid = [
     * ["1","1","1","1","0"],
     * ["1","1","0","1","0"],
     * ["1","1","0","0","0"],
     * ["0","0","0","0","0"]
     * ]
     * 输出：1
     * 示例 2：
     * 输入：grid = [
     * ["1","1","0","0","0"],
     * ["1","1","0","0","0"],
     * ["0","0","1","0","0"],
     * ["0","0","0","1","1"]
     * ]
     * 输出：3
     *
     * 提示：
     * m == grid.length
     * n == grid[i].length
     * 1 <= m, n <= 300
     * grid[i][j] 的值为 '0' 或 '1'
     */
    public static int numIslands(char[][] grid) {
        int m = grid.length, n = grid[0].length, res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    continue;
                }
                ++res;
                dfs(i, j, grid);
            }
        }
        return res;
    }

    static int[][] road = new int[][] {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    private static void dfs(int i, int j, char[][] grid) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length) {
            return;
        }
        if (grid[i][j] == 0) {
            return;
        }
        grid[i][j] = 0;
        for (int k = 0; k < road.length; k++) {
            dfs(i + road[k][0], j + road[k][1], grid);
        }
    }

    /**
     * 547 省份数量
     *
     * 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
     * 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
     * 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
     * 返回矩阵中 省份 的数量。
     *
     * 示例 1：
     * 输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
     * 输出：2
     * 示例 2：
     * 输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
     * 输出：3
     *
     * 提示：
     * 1 <= n <= 200
     * n == isConnected.length
     * n == isConnected[i].length
     * isConnected[i][j] 为 1 或 0
     * isConnected[i][i] == 1
     * isConnected[i][j] == isConnected[j][i]
     */
    public int findCircleNum(int[][] isConnected) {
        int m = isConnected.length, res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                if (isConnected[i][j] == 0) {
                    continue;
                }
                ++res;
                dfsFind(i, j, m, isConnected);
            }
        }

        return res;
    }

    private void dfsFind(int i, int j, int len, int[][] isConnected) {
        if (isConnected[i][j] == 0) {
            return;
        }
        isConnected[i][j] = 0;
        for (int k = 0; k < len; k++) {
            if (isConnected[i][k] == 1) {
                isConnected[i][k] = 0;
                dfsFind(k, i, len, isConnected);
            }
        }
    }

    /**
     * 因为省份和省份是两两互相连接的，所以可以将判断二维数组的每一个值简化为仅判断一维数组
     */
    public int findCircleNum2(int[][] isConnected) {
        int m = isConnected.length, res = 0;
        boolean[] isProvince = new boolean[m];
        for (int i = 0; i < m; i++) {
            if (!isProvince[i]) {
                ++res;
                isProvince[i] = true;
                dfsFind2(i, m, isConnected, isProvince);
            }
        }

        return res;
    }

    private void dfsFind2(int j, int len, int[][] isConnected, boolean[] isProvince) {
        for (int i = 0; i < len; i++) {
            if (!isProvince[i] && isConnected[j][i] == 1) {
                isProvince[i] = true;
                dfsFind2(i, len, isConnected, isProvince);
            }
        }
    }

    /**
     * 117
     *
     * 基于116升级为非完美二叉树
     */
    public Node connect3(Node root) {
        if (root == null) {
            return null;
        }

        Queue<Node> curr = new LinkedList<>();
        Queue<Node> next = new LinkedList<>();
        curr.offer(root);

        while (curr.size() > 0) {
            Node node = curr.poll();
            if (node.left != null) {
                next.offer(node.left);
            }
            if (node.right != null) {
                next.offer(node.right);
            }
            if (curr.size() == 0) {
                node.next = null;
                curr = next;
                next = new LinkedList<>();
            } else {
                node.next = curr.peek();
            }
        }

        return root;
    }

    Node last = null, nextStart = null;

    public Node connect4(Node root) {
        if (root == null) {
            return null;
        }
        Node start = root;
        while (start != null) {
            last = null;
            nextStart = null;
            for (Node p = start; p != null; p = p.next) {
                if (p.left != null) {
                    handle(p.left);
                }
                if (p.right != null) {
                    handle(p.right);
                }
            }
            start = nextStart;
        }
        return root;
    }

    public void handle(Node p) {
        if (last != null) {
            last.next = p;
        }
        if (nextStart == null) {
            nextStart = p;
        }
        last = p;
    }

    /**
     * 572
     * 给你两棵二叉树 root 和 subRoot 。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true ；否则，返回 false 。
     * 二叉树 tree 的一棵子树包括 tree 的某个节点和这个节点的所有后代节点。tree 也可以看做它自身的一棵子树。
     *
     * 示例 1：
     * 输入：root = [3,4,5,1,2], subRoot = [4,1,2]
     * 输出：true
     * 示例 2：
     * 输入：root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
     * 输出：false
     *
     * 提示：
     * root 树上的节点数量范围是 [1, 2000]
     * subRoot 树上的节点数量范围是 [1, 1000]
     * -104 <= root.val <= 104
     * -104 <= subRoot.val <= 104
     */
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        return dfs(root, subRoot);
    }

    private boolean dfs(TreeNode r, TreeNode s) {
        if (r == null && s != null) {
            return false;
        }
        return check(r, s) || dfs(r.left, s) || dfs(r.right, s);
    }

    private boolean check(TreeNode r, TreeNode s) {
        if (r == null && s == null) {
            return true;
        }
        if (r == null || s == null || r.val != s.val) {
            return false;
        }
        return check(r.left, s.left) && check(r.right, s.right);
    }

    /**
     * 1091
     *
     * 给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。
     * 二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：
     * 路径途经的所有单元格都的值都是 0 。
     * 路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
     * 畅通路径的长度 是该路径途经的单元格总数。
     *
     * 示例 1：
     * 输入：grid = [[0,1],[1,0]]
     * 输出：2
     * 示例 2：
     * 输入：grid = [[0,0,0],[1,1,0],[1,1,0]]
     * 输出：4
     * 示例 3：
     * 输入：grid = [[1,0,0],[1,1,0],[1,1,0]]
     * 输出：-1
     *
     * 提示：
     * n == grid.length
     * n == grid[i].length
     * 1 <= n <= 100
     * grid[i][j] 为 0 或 1
     */
    static int minSteps = Integer.MAX_VALUE;

    static int[][] road2 = new int[][] {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

    public static int shortestPathBinaryMatrix(int[][] grid) {
        dfs(0, 0, grid.length, 0, grid);
        return minSteps == Integer.MAX_VALUE ? -1 : minSteps;
    }

    private static void dfs(int m, int n, int len, int steps, int[][] grid) {
        if (m < 0 || m >= len || n < 0 || n >= len || grid[m][n] == 1 || steps > minSteps) {
            return;
        }
        ++steps;
        grid[m][n] = 1;
        if (m == len - 1 && n == len - 1) {
            minSteps = Math.min(steps, minSteps);
            grid[m][n] = 0;
            return;
        }
        for (int[] ints : road2) {
            dfs(m + ints[0], n + ints[1], len, steps, grid);
        }
        grid[m][n] = 0;
    }

    static class Node3 {
        int x;
        int y;
        int step;

        public Node3(int start, int end, int step) {
            this.x = start;
            this.y = end;
            this.step = step;
        }
    }

    int[] dx3 = {0, 0, -1, 1, -1, 1, -1, 1};
    int[] dy3 = {-1, 1, 0, 0, -1, -1, 1, 1};

    /**
     * 因为是广度优先搜索，所以第一个走到终点的，肯定是步数最少的
     */
    public int shortestPathBinaryMatrix2(int[][] grid) {

        Node3 node = new Node3(0, 0, 2);
        Deque<Node3> queue = new ArrayDeque<>();
        queue.addLast(node);

        int n = grid.length;
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) {
            return -1;
        } else if (n <= 2) {
            return n;
        }
        while (!queue.isEmpty()) {
            Node3 cur = queue.removeFirst();
            int x = cur.x;
            int y = cur.y;
            int step = cur.step;
            for (int i = 0; i < 8; i++) {
                int newX = x + dx[i];
                int newY = y + dy[i];
                if (0 <= newX && newX < n && 0 <= newY && newY < n && grid[newX][newY] == 0) {
                    //找到终点
                    if (newX == n - 1 && newY == n - 1) {
                        return step;
                    }
                    queue.addLast(new Node3(newX, newY, step + 1));
                    grid[newX][newY] = 1; //标记已遍历过，避免重复
                }
            }
        }
        return -1;
    }

    /**
     * 130. 被围绕的区域
     *
     * 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的'O' 用 'X' 填充。
     *
     * 示例 1：
     * 输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
     * 输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
     * 解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的'O'都不会被填充为'X'。 任何不在边界上，或不与边界上的'O'相连的'O'最终都会被填充为'X'。
     * 如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
     * 示例 2：
     * 输入：board = [["X"]]
     * 输出：[["X"]]
     *
     * 提示：
     * m == board.length
     * n == board[i].length
     * 1 <= m, n <= 200
     * board[i][j] 为 'X' 或 'O'
     */
    public void solve(char[][] board) {
        int m = board.length, n = board[0].length;
        for (int i = 0; i < n; i++) {
            dfs2(0, i, board);
            dfs2(m - 1, i, board);
        }
        for (int i = 0; i < m; i++) {
            dfs2(i, 0, board);
            dfs2(i, n - 1, board);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == '1') {
                    board[i][j] = 'O';
                }
            }
        }
    }

    private void dfs2(int m, int n, char[][] board) {
        if (m < 0 || m >= board.length || n < 0 || n >= board[0].length || 'X' == board[m][n] || board[m][n] == '1') {
            return;
        }
        board[m][n] = '1';
        for (int i = 0; i < 4; i++) {
            dfs2(m + road[i][0], n + road[i][1], board);
        }
    }

    /**
     * 797. 所有可能的路径
     *
     * 给你一个有n个节点的 有向无环图（DAG），请你找出所有从节点 0到节点 n-1的路径并输出（不要求按特定顺序）
     * graph[i]是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点graph[i][j]存在一条有向边）。
     *
     * 示例 1：
     * 输入：graph = [[1,2],[3],[3],[]]
     * 输出：[[0,1,3],[0,2,3]]
     * 解释：有两条路径 0 -> 1 -> 3 和 0 -> 2 -> 3
     * 示例 2：
     * 输入：graph = [[4,3,1],[3,2,4],[3],[4],[]]
     * 输出：[[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]
     *
     * 提示：
     * n == graph.length
     * 2 <= n <= 15
     * 0 <= graph[i][j] < n
     * graph[i][j] != i（即不存在自环）
     * graph[i] 中的所有元素 互不相同
     * 保证输入为 有向无环图（DAG）
     */
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        dfs(new ArrayList<>(), 0, graph);
        return res;
    }

    private void dfs(List<Integer> list, int index, int[][] graph) {
        list.add(index);
        if (index == graph.length - 1) {
            res.add(new ArrayList<>(list));
            return;
        }
        int len = graph[index].length;
        for (int i = 0; i < len; i++) {
            dfs(list, graph[index][i], graph);
            list.remove(list.size() - 1);
        }
    }


    public static void main(String[] args) {
        int[][] grid = new int[][] {{0, 1, 0, 0, 1, 1, 0}, {1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 1, 1, 1, 1}, {0, 1, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 1, 0}};
        //[[0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,1,0,0,0,1],[0,0,1,0,0,0,0,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,1,0,1,0],[0,0,0,1,0,0,1,0]]

        System.out.println(shortestPathBinaryMatrix(grid));
    }
}