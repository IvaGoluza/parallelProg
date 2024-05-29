import mpi.MPI;
import mpi.Status;

import java.util.Objects;

public class Dinner {

    private static final int LEFT_FORK_SEARCH = 0;
    private static final int RIGHT_FORK_SEARCH = 1;
    private static final int LEFT_FORK_ARRIVED = 2;
    private static final int RIGHT_FORK_ARRIVED = 3;
    private static final boolean[] buf = new boolean[]{false};
    private static int rank;
    private static int size;
    private static Status status;
    private static Philosopher philosopher;

    private static void initPhilosopher() {
        Philosopher platon;
        if (rank == 0) {   // the first one has 2 forks
            platon = new Philosopher(rank, size - 1, 1, new Fork(), new Fork());
        } else if (rank == size - 1) {  // the last one has 0 forks
            platon = new Philosopher(rank, rank - 1, 0, null, null);
        } else {  // the rest have right fork only
            platon = new Philosopher(rank, rank - 1, rank + 1, null, new Fork());
        }
        philosopher = platon;
    }

    private static void giveLeftFork() {
        MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.BOOLEAN, philosopher.getLeftNeighbor(), LEFT_FORK_SEARCH);
        if(!philosopher.getLeftFork().isClean()) {
            // my left fork is dirty -> I'm washing the dishes and sending the FORK_ARRIVED msg
            philosopher.printMsg("Giving my left fork to philosopher " + philosopher.getLeftNeighbor() + ".");
            MPI.COMM_WORLD.Isend(buf, 0, 1, MPI.BOOLEAN, philosopher.getLeftNeighbor(), RIGHT_FORK_ARRIVED);   // my left fork is his right fork
            philosopher.setLeftFork(null); // I no longer have a left fork
        } else {  // left fork is clean -> be selfish and eat first, then give it away later
            philosopher.printMsg("I will give my left fork to " + philosopher.getLeftNeighbor() + " after I eat.");
            philosopher.addRequest("L");
        }
    }

    private static void giveRightFork() {
        MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.BOOLEAN, philosopher.getRightNeighbor(), RIGHT_FORK_SEARCH);
        if(!philosopher.getRightFork().isClean()) {
            // my right fork is dirty -> I'm washing the dishes and sending the FORK_ARRIVED msg
            philosopher.printMsg("Giving my right fork to philosopher " + philosopher.getRightNeighbor() + ".");
            MPI.COMM_WORLD.Isend(buf, 0, 1, MPI.BOOLEAN, philosopher.getRightNeighbor(), LEFT_FORK_ARRIVED);  // my right fork is his left fork
            philosopher.setRightFork(null); // I no longer have a right fork
        } else {  // right fork is clean -> be selfish and eat first, then give it away later
            philosopher.printMsg("I will give my right fork to " + philosopher.getRightNeighbor() + " after I eat.");
            philosopher.addRequest("R");
        }
    }

    private static void checkRequests() {
        status = MPI.COMM_WORLD.Iprobe(philosopher.getLeftNeighbor(), LEFT_FORK_SEARCH);
        if(status != null && status.tag == LEFT_FORK_SEARCH && status.source == philosopher.getLeftNeighbor() && philosopher.getLeftFork() != null) giveLeftFork();
        status = MPI.COMM_WORLD.Iprobe(philosopher.getRightNeighbor(), RIGHT_FORK_SEARCH);
        if(status != null && status.tag == RIGHT_FORK_SEARCH && status.source == philosopher.getRightNeighbor() && philosopher.getRightFork() != null) giveRightFork();
    }

    private static void think() throws InterruptedException {
        philosopher.printMsg("Going on a thinking spree.");
        for(int i = 0; i < philosopher.getThinkingSeconds()*2; i++) {
            checkRequests();
            Thread.sleep(500);
        }
    }

    private static void leftForkHunt() {
        if(philosopher.getLeftFork() == null) {
            philosopher.printMsg("Does not have LEFT fork. Politely asking left neighbor " + philosopher.getLeftNeighbor() +" for his right fork.");
            MPI.COMM_WORLD.Isend(buf, 0, 1, MPI.BOOLEAN, philosopher.getLeftNeighbor(), RIGHT_FORK_SEARCH);
        }
        do{
            checkRequests(); // checking for all request messages
            status = MPI.COMM_WORLD.Iprobe(philosopher.getLeftNeighbor(), LEFT_FORK_ARRIVED); // did my left fork arrive
            if(status != null && status.tag == LEFT_FORK_ARRIVED && status.source == philosopher.getLeftNeighbor()) {
                MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.BOOLEAN, philosopher.getLeftNeighbor(), LEFT_FORK_ARRIVED);
                philosopher.printMsg("I got my left fork from philosopher " + philosopher.getLeftNeighbor());
                philosopher.setLeftFork(new Fork());
                philosopher.getLeftFork().washTheDishes(); // my polite neighbor has 'washed' the fork before sending it, so I know it is clean
            }
        }while(philosopher.getLeftFork() == null);
    }

    private static void rightForkHunt() {
        if(philosopher.getRightFork() == null) {
            philosopher.printMsg("Does not have RIGHT fork. Politely asking right neighbor " + philosopher.getRightNeighbor() +" for his left fork.");
            MPI.COMM_WORLD.Isend(buf, 0, 1, MPI.BOOLEAN, philosopher.getRightNeighbor(), LEFT_FORK_SEARCH);
        }
        do{
            checkRequests(); // checking for all request messages
            status = MPI.COMM_WORLD.Iprobe(philosopher.getRightNeighbor(), RIGHT_FORK_ARRIVED); // did my right fork arrive
            if(status != null && status.tag == RIGHT_FORK_ARRIVED && status.source == philosopher.getRightNeighbor()) {
                MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.BOOLEAN, philosopher.getRightNeighbor(), RIGHT_FORK_ARRIVED);
                philosopher.printMsg("I got my right fork from philosopher " + philosopher.getRightNeighbor());
                philosopher.setRightFork(new Fork());
                philosopher.getRightFork().washTheDishes(); // my polite neighbor has 'washed' the fork before sending it, so I know it is clean
            }
        }while(philosopher.getRightFork() == null);
    }

    private static void eat() throws InterruptedException {
        philosopher.printMsg("I have 2 forks. It's a feast time.");
        Thread.sleep(philosopher.getEatingSeconds() * 1000L); // eating. do not disturb
        philosopher.getLeftFork().useTheFork();
        philosopher.getRightFork().useTheFork();    // the forks are no longer clean
    }

    private static void checkRequestsQueue() {
        while(!philosopher.getRequests().isEmpty()) {
            String whichFork = philosopher.getRequests().poll();

            if(Objects.equals(whichFork, "R")) {  // have to give away my right fork

                status = MPI.COMM_WORLD.Iprobe(philosopher.getRightNeighbor(), RIGHT_FORK_SEARCH);
                if(status != null && status.tag == RIGHT_FORK_SEARCH && status.source == philosopher.getRightNeighbor())
                    MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.BOOLEAN, philosopher.getRightNeighbor(), RIGHT_FORK_SEARCH); // receive the right fork request message
                philosopher.printMsg("Giving my right fork to philosopher " + philosopher.getRightNeighbor() + ".");
                MPI.COMM_WORLD.Isend(buf, 0, 1, MPI.BOOLEAN, philosopher.getRightNeighbor(), LEFT_FORK_ARRIVED); // my right fork is his left fork
                philosopher.setRightFork(null);

            } else if(Objects.equals(whichFork, "L")) {

                status = MPI.COMM_WORLD.Iprobe(philosopher.getLeftNeighbor(), LEFT_FORK_SEARCH);
                if(status != null && status.tag == LEFT_FORK_SEARCH && status.source == philosopher.getLeftNeighbor())
                   MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.BOOLEAN, philosopher.getLeftNeighbor(), LEFT_FORK_SEARCH); // receive the left fork request message
                philosopher.printMsg("Giving my left fork to philosopher " + philosopher.getLeftNeighbor() + ".");
                MPI.COMM_WORLD.Isend(buf, 0, 1, MPI.BOOLEAN, philosopher.getLeftNeighbor(), RIGHT_FORK_ARRIVED); // my left fork is his right fork
                philosopher.setLeftFork(null);
            }
        }
    }


    public static void main(String[] args) throws InterruptedException {

        MPI.Init(args);
        Runtime.getRuntime().addShutdownHook(new Thread(MPI::Finalize));
        rank = MPI.COMM_WORLD.Rank();
        size = MPI.COMM_WORLD.Size();
        initPhilosopher();

        while(true) {
            think();
            while(!philosopher.has2Forks()) {
                if(philosopher.getRightFork() == null) rightForkHunt();
                if(philosopher.getLeftFork() == null) leftForkHunt();
            }
            eat();
            checkRequestsQueue();
        }

    }

}
