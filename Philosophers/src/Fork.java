public class Fork {
    private boolean isClean;

    public Fork() {
        this.isClean = false;
    }

    public void useTheFork() {
        this.isClean = false;
    }

    public void washTheDishes() {
        this.isClean = true;
    }

    public boolean isClean() {
        return isClean;
    }
}
