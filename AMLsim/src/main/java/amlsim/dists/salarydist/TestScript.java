package amlsim.dists.salarydist;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TestScript {

    public static void main(String[] args) {
        SalaryDistribution distribution = new SalaryDistribution();

        int sampleSize = 1000000;  // Sample 10,000 times, adjust as needed
        List<Double> samples = new ArrayList<>();

        for (int i = 0; i < sampleSize; i++) {
            samples.add(distribution.sample());
        }

        // Convert samples to a primitive array
        double[] primitiveSamples = samples.stream().mapToDouble(Double::doubleValue).toArray();

        HistogramDataset dataset = new HistogramDataset();
        dataset.addSeries("Salary Distribution", primitiveSamples, 100);  // 50 bins, adjust as needed

        JFreeChart histogram = ChartFactory.createHistogram(
                "Salary Distribution",
                "Salary",
                "Frequency",
                dataset
        );

        // Save the histogram as a PNG image
        File outputFile = new File("salary_distribution.png");
        try {
            ChartUtils.saveChartAsPNG(outputFile, histogram, 800, 600);  // Width: 800, Height: 600, adjust as needed
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Histogram saved to: " + outputFile.getAbsolutePath());
    }
}