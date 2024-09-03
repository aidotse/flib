package amlsim.dists.salarydist;

import org.apache.commons.math3.distribution.AbstractRealDistribution;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/*
 * Distribution based on statistics from SCB 
 */

public class SalaryDist extends AbstractRealDistribution {

    private static final long serialVersionUID = 1L;

    
    private class SalaryInfo {
        private int age;
        private double averageYearIncome;
        private double medianYearIncome;
        private int populationSize;
        
        public SalaryInfo() {}

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
    
    private List<SalaryInfo> salaryInfos;
    
    public SalaryDist(String csvFilePath) throws IOException {
        super(null);
        salaryInfos = new ArrayList<>();
        readCSV(csvFilePath);
    }

    private void readCSV(String csvFilePath) throws IOException {
        try (CSVReader reader = new CSVReader(new FileReader(csvFilePath))) {
            String[] nextLine;
            reader.readNext(); // skip header
            while ((nextLine = reader.readNext()) != null) {
                SalaryInfo info = new SalaryInfo();
                info.setAge(Integer.parseInt(nextLine[0]));
                info.setAverageYearIncome(Double.valueOf(nextLine[1]));
                info.setMedianYearIncome(Double.valueOf(nextLine[2]));
                int population = (int) Math.round(Double.valueOf(nextLine[3]));
                info.setPopulationSize(population);
                salaryInfos.add(info);
            }
        } catch (CsvValidationException e) {
            // You can either re-throw this, or handle it as appropriate for your application
            throw new IOException("Error validating the CSV data", e);
        }
    }

    @Override
    public double density(double x) {
        throw new UnsupportedOperationException("Density function not implemented");
    }

    @Override
    public double cumulativeProbability(double x) {
        throw new UnsupportedOperationException("cumulativeProbability function not implemented");
    }

    @Override
    public double getNumericalMean() {
        throw new UnsupportedOperationException("getNumericalMean function not implemented");
    }

    @Override
    public double getNumericalVariance() {
        throw new UnsupportedOperationException("getNumericalVariance function not implemented");
    }

    @Override
    public double getSupportLowerBound() {
        throw new UnsupportedOperationException("getSupportLowerBound function not implemented");
    }

    @Override
    public double getSupportUpperBound() {
        throw new UnsupportedOperationException("getSupportUpperBound function not implemented");
    }

    @Override
    public boolean isSupportLowerBoundInclusive() {
        throw new UnsupportedOperationException("isSupportLowerBoundInclusive function not implemented");
    }

    @Override
    public boolean isSupportUpperBoundInclusive() {
        throw new UnsupportedOperationException("isSupportUpperBoundInclusive function not implemented");
    }

    @Override
    public boolean isSupportConnected() {
        throw new UnsupportedOperationException("isSupportConnected function not implemented");
    }

    @Override
    public double sample() {
        // Create a stream of indices, weighted by population size
        int[] indices = IntStream.range(0, salaryInfos.size())
                                 .flatMap(i -> IntStream.range(0, salaryInfos.get(i).getPopulationSize()).map(j -> i))
                                 .toArray();

        // Randomly select an index
        int randomIndex = indices[ThreadLocalRandom.current().nextInt(indices.length)];

        // Here, I'm returning the average year income, but you can adjust as needed.
        return salaryInfos.get(randomIndex).getAverageYearIncome();
    }
}