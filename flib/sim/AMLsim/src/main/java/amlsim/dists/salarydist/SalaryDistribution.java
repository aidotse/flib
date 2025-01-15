package amlsim.dists.salarydist;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class SalaryDistribution {

    private List<SalaryInfo> salaryInfos;
    private double[] cumulativeProbabilities;

    public SalaryDistribution() {
        salaryInfos = new ArrayList<>();
        try {
            readCSV("src/main/java/amlsim/dists/salarydist/scb_statistics_2021.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
        computeCumulativeProbabilities();
    }

    private void readCSV(String csvFilePath) throws IOException {
        try (CSVReader reader = new CSVReader(new FileReader(csvFilePath))) {
            String[] nextLine;
            reader.readNext(); // skip header
            while ((nextLine = reader.readNext()) != null) {
                SalaryInfo info = new SalaryInfo(
                        Integer.parseInt(nextLine[0]),
                        Double.parseDouble(nextLine[1]),
                        Double.parseDouble(nextLine[2]),
                        (int) Math.round(Double.parseDouble(nextLine[3]))
                );
                salaryInfos.add(info);
            }
        } catch (CsvValidationException e) {
            throw new IOException("Error validating the CSV data", e);
        }
    }

    private void computeCumulativeProbabilities() {
        int totalPopulation = salaryInfos.stream().mapToInt(SalaryInfo::getPopulationSize).sum();
        cumulativeProbabilities = new double[salaryInfos.size()];

        int runningTotal = 0;
        for (int i = 0; i < salaryInfos.size(); i++) {
            runningTotal += salaryInfos.get(i).getPopulationSize();
            cumulativeProbabilities[i] = (double) runningTotal / totalPopulation;
        }
    }

    public double sample() {
        double randomValue = ThreadLocalRandom.current().nextDouble();
        int ageIndex = -1;

        for (int i = 0; i < cumulativeProbabilities.length; i++) {
            if (randomValue <= cumulativeProbabilities[i]) {
                ageIndex = i;
                break;
            }
        }

        if (ageIndex == -1) {
            throw new RuntimeException("Failed to sample from the cumulative distribution.");
        }

        double mean = salaryInfos.get(ageIndex).getAverageYearIncome();
        double median = salaryInfos.get(ageIndex).getMedianYearIncome();
        double mu = Math.log(median);
        double sigma = Math.sqrt(2 * Math.abs(Math.log(mean) - mu));
        
        JDKRandomGenerator rng = new JDKRandomGenerator();
        //NormalDistribution normal = new NormalDistribution(rng, mean, Math.sqrt(variance));
        LogNormalDistribution normal = new LogNormalDistribution(rng, mu, sigma);
        double salary = normal.sample() * 1000 / 12;
        
        return salary;
    }
    // SalaryInfo class remains the same...
}