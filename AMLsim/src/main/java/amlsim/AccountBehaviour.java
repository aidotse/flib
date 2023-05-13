package amlsim;

import amlsim.dists.TruncatedNormal;
import amlsim.dists.TruncatedNormalQuick;

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

    static final double lb_phone = 1.0; // should be at least 1 day until change of phone
    static final double ub_phone = 365.0 * 10.0; // have to change phone within 10 years (3650 days)
    static final double lb_bank = 0.0;
    static final double ub_bank = 365.0 * 10.0; // 10 years

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
        this.daysUntilPhoneChange = this.sampleDaysUntilNextPhoneChange();
    }

    public void updateParameters(Boolean isSAR) {
        if (isSAR) {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequencySAR();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequencySAR();
        } else {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequency();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequency();
        }
    }

    public void update() {
        // if bank change, reset the days at bank and reset number of phone changes
        if (this.daysUntilBankChange == 0) {
            this.daysUntilBankChange = this.sampleDaysUntilBankChange();
            this.numberOfPhoneChanges = 0;
        } else {
            // if no bank change, just count down days until phone change
            this.daysUntilBankChange--;

            this.daysUntilPhoneChange--;
            if (this.daysUntilPhoneChange == 0) {
                this.numberOfPhoneChanges++;
                this.daysUntilPhoneChange = this.sampleDaysUntilNextPhoneChange();
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
}
