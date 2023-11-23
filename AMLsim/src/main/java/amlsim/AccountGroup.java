package amlsim;

import amlsim.model.AbstractTransactionModel;

import java.util.*;

public class AccountGroup {

    private long accountGroupId;
    private int scheduleID;
    private long startStep;
    private long endStep;
    private int interval;
    private List<Account> members; // Accounts involved in this alert
    private Account mainAccount; // Main account of this alert
    private AbstractTransactionModel model; // Transaction model
    private AMLSim amlsim; // AMLSim main object

    AccountGroup(long accountGroupId, long startStep, long endStep, int scheduleID, int interval, AMLSim sim) {
        this.accountGroupId = accountGroupId;

        assert startStep < endStep : "startStep must be smaller than endStep";
        assert interval > 0 : "interval must be positive";

        long s1 = startStep + AMLSim.getRandom().nextInt((int) (endStep - startStep + 1));
        long s2 = startStep + AMLSim.getRandom().nextInt((int) (endStep - startStep + 1));
        this.startStep = Math.min(s1, s2);
        this.endStep = Math.max(s1, s2);
        if (this.startStep < startStep) {
            this.startStep = startStep;
        }
        if (this.endStep > endStep) {
            this.endStep = endStep;
        }

        if (scheduleID == 2) {
            this.startStep = startStep;
            this.endStep = endStep;
        }

        this.scheduleID = scheduleID;
        this.interval = interval;
        this.members = new ArrayList<>();
        this.mainAccount = null;
        this.amlsim = sim;
    }

    void setModel(AbstractTransactionModel model) {
        this.model = model;
    }

    /**
     * Add transactions
     * 
     * @param step Current simulation step
     */
    void registerTransactions(long step, Account acct) {
        // maybe add is valid step.
        model.sendTransactions(step, acct);
    }

    /**
     * Involve an account in this alert
     * 
     * @param acct Account object
     */
    void addMember(Account acct) {
        this.members.add(acct);
    }

    /**
     * Get main AMLSim object
     * 
     * @return AMLSim object
     */
    public AMLSim getSimulator() {
        return amlsim;
    }

    public int getScheduleID() {
        return this.scheduleID;
    }

    public long getStartStep() {
        return this.startStep;
    }

    public long getEndStep() {
        return this.endStep;
    }

    public int getInterval() {
        return this.interval;
    }

    /**
     * Get account group identifier as long type
     * 
     * @return Account group identifier
     */
    public long getAccoutGroupId() {
        return this.accountGroupId;
    }

    /**
     * Get member list of the alert
     * 
     * @return Alert account list
     */
    public List<Account> getMembers() {
        return members;
    }

    /**
     * Get the main account
     * 
     * @return The main account if exists.
     */
    public Account getMainAccount() {
        return mainAccount;
    }

    /**
     * Set the main account
     * 
     * @param account Main account object
     */
    public void setMainAccount(Account account) {
        this.mainAccount = account;
    }

    public AbstractTransactionModel getModel() {
        return model;
    }

    public boolean isSAR() {
        return this.mainAccount != null && this.mainAccount.isSAR();
    }
}
