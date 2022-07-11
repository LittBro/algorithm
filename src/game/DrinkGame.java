package game;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;


public class DrinkGame {

    public static void main(String[] args) throws ExecutionException, InterruptedException {

        int times = 1000000;
        int i = 0;
        Map<String, Integer> map = new HashMap();

        // 统计出单人骰型的分布
        while (i < times) {
            String type = calType();
            Integer count = map.getOrDefault(type, 0);
            map.put(type, ++count);
            ++i;
        }

        for (Map.Entry<String, Integer> set : map.entrySet()) {
            BigDecimal percent = new BigDecimal(set.getValue() * 100).divide(new BigDecimal(times), 1, RoundingMode.HALF_UP);
            System.out.println(set.getKey() + " : " + percent + "%");
        }

    }

    /**
     * 计算单人骰型的分布
     */
    static String calType() {
        Random random = new Random();
        Integer[] self;
        int maxValue;
        int secondMaxValue;
        do {
            self = new Integer[] {0, 0, 0, 0, 0, 0};
            maxValue = 0;
            secondMaxValue = 0;
            for (int i = 0; i < 5; i++) {
                int num = random.nextInt(6);
                self[num] = self[num] + 1;
                if (num != 0) {
                    if (self[num] > maxValue) {
                        maxValue = self[num];
                    } else if (self[num] > secondMaxValue) {
                        secondMaxValue = self[num];
                    }
                }
            }
        } while (self[0] <= 1 && maxValue <= 1);

        return maxValue + "_" + secondMaxValue + "_" + self[0];
    }


    /**
     * 计算出最大骰子飞/斋的概率
     *
     * @return
     */
    static Integer calculate2() {
        Random random = new Random();
        Integer[] self;
        int maxKey;
        int maxValue;
        int secondMaxValue;
        do {
            self = new Integer[] {0, 0, 0, 0, 0, 0};
            maxKey = 0;
            maxValue = 0;
            secondMaxValue = 0;
            for (int i = 0; i < 5; i++) {
                int num = random.nextInt(6);
                self[num] = self[num] + 1;
                if (self[num] > maxValue) {
                    maxKey = num;
                    maxValue = self[num];
                } else if (self[num] > secondMaxValue) {
                    secondMaxValue = self[num];
                }
            }
        } while (maxValue <= 1);

        //最大飞
        if (maxKey == 0) {
            return maxValue + secondMaxValue;
        } else {
            return self[0] + maxValue;
        }

        //最大斋
//        boolean isOne = maxKey == 0;
//        return maxValue + "";
    }
}


class Shake implements Callable {

    @Override
    public String call() {
        return calculate2();
    }

    /**
     * 计算出某种骰子飞的概率
     */
    String calculate3() {
        Random random = new Random();
        Integer[] self;
        int maxKey;
        int maxValue;
        do {
            self = new Integer[] {0, 0, 0, 0, 0, 0};
            maxKey = 0;
            maxValue = 0;
            for (int i = 0; i < 5; i++) {
                int num = random.nextInt(6);
                self[num] = self[num] + 1;
                if (self[num] > maxValue) {
                    maxKey = num;
                    maxValue = self[num];
                }
            }
        } while (maxValue <= 1);

        return self[1] + "";
    }

    /**
     * 计算出最大骰子飞/斋的概率
     *
     * @return
     */
    String calculate2() {
        Random random = new Random();
        Integer[] self;
        int maxKey;
        int maxValue;
        int secondMaxValue;
        do {
            self = new Integer[] {0, 0, 0, 0, 0, 0};
            maxKey = 0;
            maxValue = 0;
            secondMaxValue = 0;
            for (int i = 0; i < 5; i++) {
                int num = random.nextInt(6);
                self[num] = self[num] + 1;
                if (self[num] > maxValue) {
                    maxKey = num;
                    maxValue = self[num];
                } else if (self[num] > secondMaxValue) {
                    secondMaxValue = self[num];
                }
            }
        } while (maxValue <= 1);

        //最大飞
//        if (maxKey == 0) {
//            return maxValue + secondMaxValue + "";
//        } else {
//            return self[0] + maxValue + "";
//        }

        //最大斋
        boolean isOne = maxKey == 0;
        return maxValue + "";
    }

    /**
     * 计算双方的骰子
     *
     * @return
     */
    String calculate() {

        Random random = new Random();
        Integer[] self;
        Integer[] opponent;
        int maxKey;
        int maxValue;
        int opponentMaxKey;
        int opponentMaxValue;
        do {
            self = new Integer[] {0, 0, 0, 0, 0, 0};
            maxKey = 1;
            maxValue = 1;
            for (int i = 0; i < 5; i++) {
                int num = random.nextInt(6);
                self[num] = self[num] + 1;
                if (self[num] > maxValue) {
                    maxKey = num;
                    maxValue = self[num];
                }
            }
        } while (maxValue <= 1);

        do {
            opponent = new Integer[] {0, 0, 0, 0, 0, 0};
            opponentMaxKey = 1;
            opponentMaxValue = 1;
            for (int i = 0; i < 5; i++) {
                int num = random.nextInt(6);
                opponent[num] = opponent[num] + 1;
                if (opponent[num] > opponentMaxValue) {
                    opponentMaxKey = num;
                    opponentMaxValue = opponent[num];
                }
            }
        } while (opponentMaxValue <= 1);

//        return maxSize + "_" + secondSize + "_" + nums[0];
//        return nums[1] + "";


        boolean isEqual = opponentMaxKey == maxKey;
        int selfMaxTotal = maxValue + opponent[maxKey];
        int opponentMaxTotal = opponentMaxValue + self[opponentMaxKey];
//        return "己方摇出最多的骰子数量：" + maxValue + "，骰子是：" + maxKey + "，合计结果为：" + selfMaxTotal;
//        return "己方摇出最多的骰子数量：" + maxValue + "，合计结果为：" + selfMaxTotal;
        return maxValue + "_" + selfMaxTotal;
    }
}