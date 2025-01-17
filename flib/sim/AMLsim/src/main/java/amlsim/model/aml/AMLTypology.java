
// Source: http://files.acams.org/pdfs/English_Study_Guide/Chapter_5.pdf
// Suspicious transactions which may lead to money laundering
// - Unusually high monthly balances in comparison to known sources of income.
// - Unusually large deposits, deposits in round numbers or deposits in repeated amounts that are not attributable to legitimate sources of income.
// - Multiple deposits made under reportable thresholds.
// - The timing of deposits. This is particularly important when dates of illegal payments are known.
// - Checks written for unusually large amounts (in relation to the suspect's known practices).
// - A lack of account activity. This might indicate transactions in currency or the existence of other unknown bank accounts.
//
// Note: No specific bank models are used for this AML typology model class.

package amlsim.model.aml;

import amlsim.AMLSim;
import amlsim.Account;
import amlsim.Alert;
import amlsim.model.AbstractTransactionModel;

/**
 * Suspicious transaction models
 */
public abstract class AMLTypology extends AbstractTransactionModel {

    // Transaction model ID of AML typologies
    protected static final int AML_FAN_OUT = 1;
    protected static final int AML_FAN_IN = 2;
    protected static final int CYCLE = 3;
    protected static final int BIPARTITE = 4;
    protected static final int STACK = 5;
    protected static final int RANDOM = 6;
    protected static final int SCATTER_GATHER = 7; // fan-out -> fan-in
    protected static final int GATHER_SCATTER = 8; // fan-in -> fan-out

    // Transaction scheduling ID
    static final int FIXED_INTERVAL = 0; // All accounts send money in order with the same interval
    static final int RANDOM_INTERVAL = 1; // All accounts send money in order with random intervals
    static final int UNORDERED = 2; // All accounts send money randomly
    static final int SIMULTANEOUS = 3; // All transactions are performed at single step simultaneously

    final double marginRatio = AMLSim.getSimProp().getMarginRatio(); // Each member holds this ratio of the received amount

    protected long stepReciveFunds;
    protected String sourceType;
    
    /**
     * Create an AML typology object (alert transaction model)
     * 
     * @param modelID   Alert transaction model ID as int
     * @param minAmount Minimum transaction amount
     * @param maxAmount Maximum transaction amount
     * @param startStep Start simulation step (all transactions will start after
     *                  this step)
     * @param endStep   End simulation step (all transactions will finish before
     *                  this step)
     * @return AML typology model object
     */
    public static AMLTypology createTypology(int modelID, double minAmount, double maxAmount, int startStep, int endStep, int scheduleID, int interval, String sourceType) {
        AMLTypology model;
        switch (modelID) {
            case AML_FAN_OUT:
                model = new FanOutTypology(minAmount, maxAmount, startStep, endStep, scheduleID, interval, sourceType);
                break;
            case AML_FAN_IN:
                model = new FanInTypology(minAmount, maxAmount, startStep, endStep, scheduleID, interval, sourceType);
                break;
            case CYCLE:
                model = new CycleTypology(minAmount, maxAmount, startStep, endStep, scheduleID, interval, sourceType);
                break;
            case BIPARTITE:
                model = new BipartiteTypology(minAmount, maxAmount, startStep, endStep, scheduleID, interval, sourceType);
                break;
            case STACK:
                model = new StackTypology(minAmount, maxAmount, startStep, endStep, scheduleID, interval, sourceType);
                break;
            case RANDOM:
                model = new RandomTypology(minAmount, maxAmount, startStep, endStep, scheduleID, interval, sourceType);
                break;
            case SCATTER_GATHER:
                model = new ScatterGatherTypology(minAmount, maxAmount, startStep, scheduleID, interval, endStep, sourceType);
                break;
            case GATHER_SCATTER:
                model = new GatherScatterTypology(minAmount, maxAmount, startStep, scheduleID, interval, endStep, sourceType);
                break;
            default:
                throw new IllegalArgumentException("Unknown typology model ID: " + modelID);
        }
        model.setParameters(startStep, endStep); // TODO: remove?
        return model;
    }

    Alert alert;
    protected double minAmount;
    protected double maxAmount;
    protected long startStep;
    protected long endStep;

    /**
     * Set parameters (timestamps and amounts) of transactions
     * 
     * @param modelID Scheduling model ID
     */
    public abstract void setParameters(long seed);

    // /**
    // * Get the number of total transactions in this alert
    // * @return Number of transactions
    // */
    // public abstract int getNumTransactions();

    /**
     * Bind this alert transaction model to the alert
     * 
     * @param ag Alert object
     */
    public void setAlert(Alert ag) {
        this.alert = ag;
    }

    /**
     * Whether the current simulation step is within the valid simulation step range
     * 
     * @param step Current simulation step
     * @return If the current step is within the valid simulation step range, return
     *         true
     */
    public boolean isValidStep(long step) {
        return startStep <= step && step <= endStep;
    }

    
    /**
     * Checks if the given amount is valid for the specified account.
     * @param amount the amount to check
     * @param acct the account to check against
     * @return true if the account balance is greater than or equal to the amount, false otherwise
     */
    public boolean isValidAmount(double amount, Account acct) {
        double balance = acct.getBalance();
        return balance >= amount;
    }

    public int getStepRange() {
        return (int) (endStep - startStep + 1);
    }

    /**
     * Construct an AML typology with the given basic parameters
     * 
     * @param minAmount Minimum transaction amount (each transaction amount must not
     *                  be lower than this value)
     * @param maxAmount Maximum transaction amount (each transaction amount must not
     *                  be higher than this value)
     * @param startStep Start simulation step of alert transactions (any
     *                  transactions cannot be carried out before this step)
     * @param endStep   End simulation step of alert transactions (any transactions
     *                  cannot be carried out after this step)
     */
    public AMLTypology(double minAmount, double maxAmount, int startStep, int endStep, int scheduleID, int interval, String sourceType) {
        this.minAmount = minAmount;
        this.maxAmount = maxAmount;
        this.startStep = startStep;
        this.endStep = endStep;
        this.scheduleID = scheduleID;
        this.interval = interval;
        this.sourceType = sourceType;
        // Set step for reciving illicit funds
        if (this.sourceType.equals("TRANSFER")) {
            if (this.startStep < 25) { // TODO: change this when inital balance is set as a transaction on step 0, if ML pattern starts before step 25 we'll assume the funds are in the inital balance.  
                //this.startStep = this.startStep + 25;
                //this.endStep = this.endStep + 25;
                this.stepReciveFunds = 0;
            } else {
                this.stepReciveFunds = (long) Math.floor(this.startStep / 28) * 28; 
            }
        } else {
            this.stepReciveFunds = this.startStep; // TODO: implement some randomization, and multiple steps if several origs
        }
    }

    /**
     * Update the minimum transaction amount if the given amount is smaller than the
     * current one
     * 
     * @param minAmount New minimum amount
     */
    public void updateMinAmount(double minAmount) {
        this.minAmount = Math.min(this.minAmount, minAmount);
    }

    /**
     * Update the maximum transaction amount if the given amount is larger than the
     * current one
     * 
     * @param maxAmount New maximum amount
     */
    public void updateMaxAmount(double maxAmount) {
        this.maxAmount = Math.max(this.maxAmount, maxAmount);
    }

    public void updateStartStep(long startStep) {
        this.startStep = Math.min(this.startStep, startStep);
    }

    public void updateEndStep(long endStep) {
        this.endStep = Math.max(this.endStep, endStep);
    }

    /**
     * Generate a random amount
     * 
     * @return A random amount within "minAmount" and "maxAmount"
     */
    double getRandomAmount() {
        return alert.getSimulator().random.nextDouble() * (maxAmount - minAmount) + minAmount;
    }

    /**
     * Generate a random simulation step
     * 
     * @return Random simulation step within startStep and endStep
     */
    long getRandomStep() {
        return alert.getSimulator().random.nextLong(getStepRange()) + startStep;
    }

    /**
     * Generate a random simulation step from the given start and end steps
     * 
     * @param start Minimum simulation step
     * @param end   Maximum simulation
     * @return Random simulation step within the given step range
     */
    long getRandomStepRange(long start, long end) {
        if (start < startStep || endStep < end) {
            throw new IllegalArgumentException(
                    "The start and end steps must be within " + startStep + " and " + endStep);
        } else if (end < start) {
            throw new IllegalArgumentException("The start and end steps are unordered");
        }
        long range = end - start;
        if (range < 1) { // TODO: bug fix, the python script shouldn't generate such a case
            range = 1;
        }
        return alert.getSimulator().random.nextLong(range) + start;
    }

    @Override
    public String getModelName() {
        return "AMLTypology";
    }

    // ???
    // this is a no-op coming from SAR Account
    // this eventually should go away.
    // To Do Jordan D. Nelson
    public final void makeTransaction(long step) {
    }
}
