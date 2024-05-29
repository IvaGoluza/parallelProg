import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

public class Philosopher {
    private final int rank;
    private final int leftNeighbor;
    private final int rightNeighbor;
    private Fork leftFork;
    private Fork rightFork;
    private final Queue<String> requests;
    private final int thinkingSeconds;
    private final int eatingSeconds;


    public Philosopher(int rank, int leftNeighbor, int rightNeighbor, Fork leftFork, Fork rightFork) {
        this.rank = rank;
        this.leftNeighbor = leftNeighbor;
        this.rightNeighbor = rightNeighbor;
        this.leftFork = leftFork;
        this.rightFork = rightFork;
        this.requests = new LinkedList<>();
        Random random = new Random();
        this.thinkingSeconds = random.nextInt(7) + 1;
        this.eatingSeconds = random.nextInt(4) + 1;
    }
    public void printMsg(String msg) {
        System.out.println("\t".repeat(Math.max(0, this.rank)) + "Philosopher " + this.rank + ": " + msg);
    }
    public boolean has2Forks() {
        return leftFork != null && rightFork != null;
    }
    public int getLeftNeighbor() {
        return leftNeighbor;
    }
    public int getRightNeighbor() {
        return rightNeighbor;
    }
    public Fork getLeftFork() {
        return leftFork;
    }
    public Fork getRightFork() {
        return rightFork;
    }
    public Queue<String> getRequests() {
        return requests;
    }
    public int getThinkingSeconds() {
        return thinkingSeconds;
    }
    public int getEatingSeconds() {
        return eatingSeconds;
    }
    public void setLeftFork(Fork leftFork) {
        this.leftFork = leftFork;
    }
    public void setRightFork(Fork rightFork) {
        this.rightFork = rightFork;
    }
    public void addRequest(String request) {
        if(requests.size() <= 2) requests.add(request);  // there can only be 2 requests from neighbors
    }

}
