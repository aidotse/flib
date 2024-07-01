package amlsim;

import amlsim.dists.TruncatedNormal;
import amlsim.dists.TruncatedNormalQuick;

import java.util.*;

public class AccountBehaviour {

    private SimProperties simProperties;
    private Boolean isSAR;

    private int daysUntilPhoneChange;
    private int numberOfPhoneChanges = 0;
    private double meanPhoneChange = 0;
    private double stdPhoneChange = 0;

    private double meanBankChange = 0;
    private double stdBankChange = 0;
    private int daysUntilBankChange = 0;
    private long daysInBank = 0;

    private int daysUntilBankChangeWithoutSAR=0;
    private int daysUntilPhoneChangeWithoutSAR=0;
    private long daysInBankWithoutSAR=0;
    private int numberOfPhoneChangesWithoutSAR=0;

    static final double lb_phone = 1.0; // should be at least 1 day until change of phone
    static final double ub_phone = 365.0 * 10.0; // have to change phone within 10 years (3650 days)
    static final double lb_bank = 0.0;
    static final double ub_bank = 365.0 * 10.0; // 30 years is the upper limit to change bank

    private Random random;

    public AccountBehaviour(Boolean isSAR) {
        this.simProperties = AMLSim.getSimProp();
        this.isSAR = isSAR;

        if (this.isSAR) {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequencySAR();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequencySAR();
            this.meanBankChange = simProperties.getMeanBankChangeFrequencySAR();
            this.stdBankChange = simProperties.getStdBankChangeFrequencySAR();
        } else {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequency();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequency();
            this.meanBankChange = simProperties.getMeanBankChangeFrequency();
            this.stdBankChange = simProperties.getStdBankChangeFrequency();
        }

        this.random = new Random();
        this.daysUntilPhoneChange = this.sampleDaysUntilNextPhoneChange();
        this.daysUntilBankChange = this.sampleDaysUntilBankChange();
    }

    public void updateParameters(Boolean isSAR) {
        // This function is required since we may not know that the account is SAR or
        // not at the beginning
        if (isSAR) {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequencySAR();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequencySAR();
            this.meanBankChange = simProperties.getMeanBankChangeFrequencySAR();
            this.stdBankChange = simProperties.getStdBankChangeFrequencySAR();
        } else {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequency();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequency();
            this.meanBankChange = simProperties.getMeanBankChangeFrequency();
            this.stdBankChange = simProperties.getStdBankChangeFrequency();
        }
        // resample the days
        this.daysUntilPhoneChange = this.sampleDaysUntilNextPhoneChange();
        this.daysUntilBankChange = sampleDaysUntilBankChange();
    }


    public void update() {
        // if bank change, reset the days at bank and reset number of phone changes
        if (this.daysUntilBankChange == 0) {
            this.daysUntilBankChange = this.sampleDaysUntilBankChange();
            this.numberOfPhoneChanges = 0;
            this.daysInBank = 0;
        } else {
            // if no bank change, just count down days until phone change
            this.daysUntilBankChange--;
            this.daysInBank++;
            this.daysUntilPhoneChange--;
            if (this.daysUntilPhoneChange == 0) {
                this.numberOfPhoneChanges++;
                this.daysUntilPhoneChange = this.sampleDaysUntilNextPhoneChange();
            }
        }
    }

    public void updateWithoutSAR() {
        // if bank change, reset the days at bank and reset number of phone changes
        double meanpc = simProperties.getMeanPhoneChangeFrequency();
        double stdpc = simProperties.getStdPhoneChangeFrequency();
        double meanbc = simProperties.getMeanBankChangeFrequency();
        double stdbc = simProperties.getStdBankChangeFrequency();
        

        if (this.daysUntilBankChangeWithoutSAR == 0) {
            TruncatedNormal tn =new TruncatedNormal(meanbc, stdbc, lb_bank, ub_bank);
            this.daysUntilBankChangeWithoutSAR = (int) tn.sample();
            this.numberOfPhoneChangesWithoutSAR = 0;
            this.daysInBankWithoutSAR = 0;
        } else {
            // if no bank change, just count down days until phone change
            this.daysUntilBankChangeWithoutSAR--;
            this.daysInBankWithoutSAR++;
            this.daysUntilPhoneChangeWithoutSAR--;
            if (this.daysUntilPhoneChangeWithoutSAR == 0) {
                this.numberOfPhoneChangesWithoutSAR++;
                TruncatedNormal tn = new TruncatedNormal(meanpc, stdpc, lb_phone, ub_phone);
                this.daysUntilPhoneChangeWithoutSAR = (int) tn.sample();
            }
        }
    }

    public int sampleDaysUntilNextPhoneChange() {
        TruncatedNormal tn = new TruncatedNormal(this.meanPhoneChange, this.stdPhoneChange, lb_phone, ub_phone);
        int days = (int) tn.sample();
        return days;
    }

    public int sampleDaysUntilBankChange() {
        TruncatedNormal tn = new TruncatedNormal(this.meanBankChange, this.stdBankChange, lb_bank, ub_bank);
        int days = (int) tn.sample();
        return days;
    }

    public int getNumberOfPhoneChanges() {
        return this.numberOfPhoneChanges;
    }

    public int getNumberOfPhoneChangesWithoutSAR(){
        return this.numberOfPhoneChangesWithoutSAR;
    }
    
    public long getDaysInBank() {
        return this.daysInBank;
    }

    public String getNewBank(String currentBank) {
        if (this.daysUntilBankChange == 0) {
            // If the bank is changed, return a new bank
            List<String> bankList = new ArrayList<>();
            for (String s : AMLSim.getAvailableBanks()) {
                if (!s.equals(currentBank)) {
                    bankList.add(s);
                }
            }
            if (bankList.isEmpty()) {
                return currentBank;
            }
            String newBank = bankList.get(random.nextInt(bankList.size()));
            return newBank;
        } else {
            // If the bank is not changed, return the current bank
            return currentBank;
        }
    }
}
