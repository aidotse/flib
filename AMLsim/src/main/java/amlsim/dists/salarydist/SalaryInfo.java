package amlsim.dists.salarydist;

public class SalaryInfo {
    private int age;
    private double averageYearIncome;
    private double medianYearIncome;
    private int populationSize;
    
    public SalaryInfo(int age, double averageYearIncome, double medianYearIncome, int populationSize) {
        this.age = age;
        this.averageYearIncome = averageYearIncome;
        this.medianYearIncome = medianYearIncome;
        this.populationSize = populationSize;
    }
    
    public int getAge() {
        return age;
    }
    
    public void setAge(int age) {
        this.age = age;
    }
    
    public double getAverageYearIncome() {
        return averageYearIncome;
    }
    
    public void setAverageYearIncome(double averageYearIncome) {
        this.averageYearIncome = averageYearIncome;
    }
    
    public double getMedianYearIncome() {
        return medianYearIncome;
    }
    
    public void setMedianYearIncome(double medianYearIncome) {
        this.medianYearIncome = medianYearIncome;
    }
    
    public int getPopulationSize() {
        return populationSize;
    }
    
    public void setPopulationSize(int populationSize) {
        this.populationSize = populationSize;
    }
}
