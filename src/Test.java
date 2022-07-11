import java.time.Instant;
import java.time.LocalDate;
import java.time.Period;
import java.time.ZoneId;

public class Test {

//    private static Object a;

//    public static void main(String[] args) throws Exception {

//        synchronized (a) {
//            String str1 = "bbaacasdfcvasdfassadfasvcxv";
//        }
//        String str2 = "bbaacasdfcvasdsdfsdffassadfasvcxv";
//
//        ReentrantLock reentrantLock = new ReentrantLock();
//        reentrantLock.lock();
//        reentrantLock.unlock();
//        reentrantLock.newCondition();
//
//        AbstractQueuedSynchronizer abstractQueuedSynchronizer = new AbstractQueuedSynchronizer() {
//            @Override
//            protected boolean tryAcquire(int arg) {
//                return super.tryAcquire(arg);
//            }
//        };
//
//        LinkedHashMap<String, String> map = new LinkedHashMap<>();

//        Integer a = new Integer(10);
//        Integer b = new Integer(10);
//        System.out.println(a==b);
//        System.out.println(a.equals(b));
//        System.out.println(a==10);
//
//        System.out.println(test.class.getClassLoader());
//    }

//    /**
//     * @param string
//     * @return
//     */
//    static Long stringToLong(String string) {
//        char[] chars = string.toCharArray();
//        long result = 0;
//        boolean isNegative = false;
//        for (int i = 0; i < chars.length; i++) {
//            if (chars[i] == '-') {
//                isNegative = true;
//                continue;
//            }
//            result = result * 10;
//            long num = chars[i] - '0';
//            result += num;
//        }
//
//        return isNegative ? result * -1 : result;
//    }

    /**
     * 给定一个字符串str，请在单词间做逆序调整
     * 举例：
     * "when I see your smile , i see you" 逆序成"smile your see I when”
     *
     *
     * 要求，只能用数组作为临时存储，不使用String等
     */
    private static String reverseWord(String str) {
        if (str == null || str.length() <= 1) {
            return str;
        }
        String[] strings = str.split(" ");

        int left = 0;
        int right = strings.length - 1;

        while (left < right) {
            String temp = strings[left];
            strings[left] = strings[right];
            strings[right] = temp;
            ++left;
            --right;
        }

        StringBuilder result = new StringBuilder(strings[0]);
        for (int i = 1; i < strings.length; i++) {
            result.append(" ").append(strings[i]);
        }

        return result.toString();
    }


    public static void main(String[] args) {
        LocalDate create = LocalDate.ofInstant(Instant.ofEpochMilli(1431124298856L), ZoneId.systemDefault());

        LocalDate now = LocalDate.now();
        Period period = Period.between(create, now);
        System.out.println(period.getYears() + "_" + period.getMonths());
    }

    static void asyncPrint() {
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("aaa");
    }
}