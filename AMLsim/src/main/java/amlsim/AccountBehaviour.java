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

    static final double lb = 1.0; // should be at least 1 day until change of phone
    static final double ub = 3650.0; // have to change phone within 10 years (3650 days)

    public AccountBehaviour(Boolean isSAR) {
        this.simProperties = AMLSim.getSimProp();
        this.isSAR = isSAR;

        if (this.isSAR) {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequencySAR();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequencySAR();
        } else {
            this.meanPhoneChange = simProperties.getMeanPhoneChangeFrequency();
            this.stdPhoneChange = simProperties.getStdPhoneChangeFrequency();
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
        this.daysUntilPhoneChange--;
        if (this.daysUntilPhoneChange == 0) {
            this.numberOfPhoneChanges++;
            this.daysUntilPhoneChange = this.sampleDaysUntilNextPhoneChange();
        }
    }

    public int sampleDaysUntilNextPhoneChange() {
        TruncatedNormal tn = new TruncatedNormal(this.meanPhoneChange, this.stdPhoneChange, lb, ub);
        int days = (int) tn.sample();
        return days;
    }

    public int getNumberOfPhoneChanges() {
        return this.numberOfPhoneChanges;
    }
}
