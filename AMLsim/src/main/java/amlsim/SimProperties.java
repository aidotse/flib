package amlsim;

import java.io.*;
import java.nio.file.*;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.json.*;
import org.mockito.internal.hamcrest.MatcherGenericTypeExtractor;

/**
 * Simulation properties and global parameters loaded from the configuration
 * JSON file
 */
public class SimProperties {

    private static final String separator = File.separator;
    private JSONObject generalProp;
    private JSONObject simProp;
    private JSONObject inputProp;
    private JSONObject outputProp;
    private JSONObject cashInProp;
    private JSONObject cashOutProp;
    private String workDir;
    private double marginRatio; // Ratio of margin for AML typology transactions
    private int seed; // Seed of randomness
    private String simName; // Simulation name

    private int normalTxInterval;
    private double minTxAmount; // Minimum base (normal) transaction amount
    private double maxTxAmount; // Maximum base (suspicious) transaction amount
    private double meanTxAmount; // Mean of base transaction amount
    private double stdTxAmount; // Standard deviation of base transaction amount
    private double meanTxAmountSAR; // Mean of SAR transaction amount
    private double stdTxAmountSAR; // Standard deviation of SAR transaction amount

    // Swish-related parameters
    private double meanPhoneChangeFrequency;
    private double stdPhoneChangeFrequency;
    private double meanPhoneChangeFrequencySAR;
    private double stdPhoneChangeFrequencySAR;
    private double meanBankChangeFrequency;
    private double stdBankChangeFrequency;
    private double meanBankChangeFrequencySAR;
    private double stdBankChangeFrequencySAR;

    // Income and outcome parameters
    private double probIncome;
    private double meanIncome;
    private double stdIncome;
    private double probIncomeSAR;
    private double meanIncomeSAR;
    private double stdIncomeSAR;
    //private double probOutcome;
    //private double meanOutcome;
    //private double stdOutcome;
    //private double probOutcomeSar;
    //private double meanOutcomeSar;
    //private double stdOutcomeSar;

    SimProperties(String jsonName) throws IOException {
        String jsonStr = loadTextFile(jsonName);
        JSONObject jsonObject = new JSONObject(jsonStr);
        JSONObject defaultProp = jsonObject.getJSONObject("default");

        generalProp = jsonObject.getJSONObject("general");
        simProp = jsonObject.getJSONObject("simulator");
        inputProp = jsonObject.getJSONObject("temporal"); // Input directory of this simulator is temporal directory
        outputProp = jsonObject.getJSONObject("output");

        normalTxInterval = simProp.getInt("transaction_interval");
        minTxAmount = defaultProp.getDouble("min_amount");
        maxTxAmount = defaultProp.getDouble("max_amount");
        meanTxAmount = defaultProp.getDouble("mean_amount");
        stdTxAmount = defaultProp.getDouble("std_amount");
        meanTxAmountSAR = defaultProp.getDouble("mean_amount_sar");
        stdTxAmountSAR = defaultProp.getDouble("std_amount_sar");

        meanPhoneChangeFrequency = defaultProp.getDouble("mean_phone_change_frequency");
        stdPhoneChangeFrequency = defaultProp.getDouble("std_phone_change_frequency");
        meanPhoneChangeFrequencySAR = defaultProp.getDouble("mean_phone_change_frequency_sar");
        stdPhoneChangeFrequencySAR = defaultProp.getDouble("std_phone_change_frequency_sar");

        meanBankChangeFrequency = defaultProp.getDouble("mean_bank_change_frequency");
        stdBankChangeFrequency = defaultProp.getDouble("std_bank_change_frequency");
        meanBankChangeFrequencySAR = defaultProp.getDouble("mean_bank_change_frequency_sar");
        stdBankChangeFrequencySAR = defaultProp.getDouble("std_bank_change_frequency_sar");

        //probIncome = defaultProp.getDouble("prob_income");
        //meanIncome = defaultProp.getDouble("mean_income");
        //stdIncome = defaultProp.getDouble("std_income");
        //probIncome = defaultProp.getDouble("prob_income_sar");
        //meanIncome = defaultProp.getDouble("mean_income_sar");
        //stdIncome = defaultProp.getDouble("std_income_sar");

        System.out.printf("General transaction interval: %d\n", normalTxInterval);
        System.out.printf("Base transaction amount: Normal = %f, Suspicious= %f\n", minTxAmount, maxTxAmount);

        cashInProp = defaultProp.getJSONObject("cash_in");
        cashOutProp = defaultProp.getJSONObject("cash_out");
        marginRatio = defaultProp.getDouble("margin_ratio");

        String envSeed = System.getenv("RANDOM_SEED");
        seed = envSeed != null ? Integer.parseInt(envSeed) : generalProp.getInt("random_seed");
        System.out.println("Random seed: " + seed);

        simName = System.getProperty("simulation_name");
        if (simName == null) {
            simName = generalProp.getString("simulation_name");
        }
        System.out.println("Simulation name: " + simName);

        String simName = getSimName();
        workDir = inputProp.getString("directory") + separator + simName + separator;
        System.out.println("Working directory: " + workDir);
    }

    private static String loadTextFile(String jsonName) throws IOException {
        Path file = Paths.get(jsonName);
        byte[] bytes = Files.readAllBytes(file);
        return new String(bytes);
    }

    String getSimName() {
        return simName;
    }

    public int getSeed() {
        return seed;
    }

    public int getSteps() {
        return generalProp.getInt("total_steps");
    }

    boolean isComputeDiameter() {
        return simProp.getBoolean("compute_diameter");
    }

    int getTransactionLimit() {
        return simProp.getInt("transaction_limit");
    }

    int getNormalTransactionInterval() {
        return normalTxInterval;
    }

    public double getMinTransactionAmount() { // TODO: remove
        return minTxAmount;
    }

    public double getMaxTransactionAmount() { // TODO: remove
        return maxTxAmount;
    }

    public double getMeanTransactionAmount() {
        return meanTxAmount;
    }

    public double getStdTransactionAmount() {
        return stdTxAmount;
    }

    public double getMeanTransactionAmountSAR() {
        return meanTxAmountSAR;
    }

    public double getStdTransactionAmountSAR() {
        return stdTxAmountSAR;
    }

    public double getMeanPhoneChangeFrequency() {
        return meanPhoneChangeFrequency;
    }

    public double getStdPhoneChangeFrequency() {
        return stdPhoneChangeFrequency;
    }

    public double getMeanPhoneChangeFrequencySAR() {
        return meanPhoneChangeFrequencySAR;
    }

    public double getStdPhoneChangeFrequencySAR() {
        return stdPhoneChangeFrequencySAR;
    }

    public double getMeanBankChangeFrequency() {
        return meanBankChangeFrequency;
    }

    public double getStdBankChangeFrequency() {
        return stdBankChangeFrequency;
    }

    public double getMeanBankChangeFrequencySAR() {
        return meanBankChangeFrequencySAR;
    }

    public double getStdBankChangeFrequencySAR() {
        return stdBankChangeFrequencySAR;
    }

    public double getProbIncome() {
        return probIncome;
    }

    public double getMeanIncome() {
        return meanIncome;
    }

    public double getStdIncome() {
        return stdIncome;
    }

    public double getProbIncomeSAR() {
        return probIncomeSAR;
    }

    public double getMeanIncomeSAR() {
        return meanIncomeSAR;
    }

    public double getStdIncomeSAR() {
        return stdIncomeSAR;
    }
    
    //public double getProbOutcome() {
    //    return probOutcome;
    //}
    //
    //public double getMeanOutcome() {
    //    return meanOutcome;
    //}
    //
    //public double getStdOutcome() {
    //    return stdOutcome;
    //}

    public double getMarginRatio() {
        return marginRatio;
    }

    int getNumBranches() {
        return simProp.getInt("numBranches");
    }

    String getInputAcctFile() {
        return workDir + inputProp.getString("accounts");
    }

    String getInputTxFile() {
        return workDir + inputProp.getString("transactions");
    }

    String getInputAlertMemberFile() {
        return workDir + inputProp.getString("alert_members");
    }

    String getNormalModelsFile() {
        return workDir + inputProp.getString("normal_models");
    }

    String getOutputTxLogFile() {
        return getOutputDir() + outputProp.getString("transaction_log");
    }

    String getOutputDir() {
        return outputProp.getString("directory") + separator + simName + separator;
    }

    String getCounterLogFile() {
        return getOutputDir() + outputProp.getString("counter_log");
    }

    String getDiameterLogFile() {
        return workDir + outputProp.getString("diameter_log");
    }

    int getCashTxInterval(boolean isCashIn, boolean isSAR) {
        String key = isSAR ? "fraud_interval" : "normal_interval";
        return isCashIn ? cashInProp.getInt(key) : cashOutProp.getInt(key);
    }

    float getCashTxMinAmount(boolean isCashIn, boolean isSAR) {
        String key = isSAR ? "fraud_min_amount" : "normal_min_amount";
        return isCashIn ? cashInProp.getFloat(key) : cashOutProp.getFloat(key);
    }

    float getCashTxMaxAmount(boolean isCashIn, boolean isSAR) {
        String key = isSAR ? "fraud_max_amount" : "normal_max_amount";
        return isCashIn ? cashInProp.getFloat(key) : cashOutProp.getFloat(key);
    }
}
