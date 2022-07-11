package algorithm.interview;

public class Question {

    /**
     * 61
     * 给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动k个位置。
     *
     * 示例 1：
     * 输入：head = [1,2,3,4,5], k = 2
     * 输出：[4,5,1,2,3]
     * 示例 2：
     * 输入：head = [0,1,2], k = 4
     * 输出：[2,0,1]
     *
     * 提示：
     * 链表中节点的数目在范围 [0, 500] 内
     * -100 <= Node.val <= 100
     * 0 <= k <= 2 * 109
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        int size = 2;
        ListNode node = head.next, pre = new ListNode();
        while (node.next != null) {
            ++size;
            node = node.next;
        }
        node.next = head;
        int position = size - k % size;
        System.out.println(size + "_" + position);
        while (position > 0) {
            position--;
            pre = head;
            head = head.next;
        }
        pre.next = null;

        return head;
    }
}


class ListNode {
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
