package amlsim;

import amlsim.model.*;
import amlsim.model.cash.CashInModel;
import amlsim.model.cash.CashOutModel;
import net.bytebuddy.dynamic.scaffold.MethodGraph.Linked;
import sim.engine.SimState;
import sim.engine.Steppable;
import java.util.*;
import amlsim.dists.TruncatedNormal;
import amlsim.dists.salarydist.SalaryDistribution;
import amlsim.AccountBehaviour;

public class Account implements Steppable {

	protected String id;

	protected CashInModel cashInModel;
	protected CashOutModel cashOutModel;
	protected boolean isSAR = false;
	private Branch branch = null;
	private Set<String> origAcctIDs = new HashSet<>(); // Originator account ID set
	private Set<String> beneAcctIDs = new HashSet<>(); // Beneficiary account ID set
	private List<Account> origAccts = new ArrayList<>(); // Originator accounts from which this account receives money
	private List<Account> beneAccts = new ArrayList<>(); // Beneficiary accounts to which this account sends money
	private int numSARBene = 0; // Number of SAR beneficiary accounts
	private String bankID = ""; // Bank ID

	private double monthlyIncome = 0; // Salary
	private double monthlyIncomeSar = 0; //
	private double monthlyOutcome = 0; // Rent
	private double monthlyOutcomeSar = 0; // Rent
	private int stepMonthlyOutcome = 26; // Step of monthly outcome

	private double probIncome; // Probability of income
	private double meanIncome; // Mean income
	private double stdIncome; // Standard deviation of income
	private double probIncomeSar; // Probability of income
	private double meanIncomeSar; // Mean income
	private double stdIncomeSar; // Standard deviation of income
	private double meanOutcome; // Mean outcome
	private double stdOutcome; // Standard deviation of outcome
	private double meanOutcomeSar; // Mean outcome
	private double stdOutcomeSar; // Standard deviation of outcome
	private double probSpendCash; // Probability of spending cash

	List<Alert> alerts = new ArrayList<>();
	List<AccountGroup> accountGroups = new ArrayList<>();

	private Map<String, String> tx_types = new HashMap<>(); // Receiver Client ID --> Transaction Type

	private static List<String> all_tx_types = new ArrayList<>();

	private ArrayList<String> paramFile = new ArrayList<>();

	private double balance = 0;
	private double cashBalance = 0;

	protected long startStep = 0;
	protected long endStep = 0;

	private Random random;

	private AccountBehaviour accountBehaviour;

	private LinkedList<Double> balanceHistory = new LinkedList<Double>();

	public Account() {
		this.id = "-";
	}

	/**
	 * Constructor of the account object
	 * 
	 * @param id          Account ID
	 * @param interval    Default transaction interval
	 * @param initBalance Initial account balance
	 * @param start       Start step
	 * @param end         End step
	 */
	public Account(String id, int interval, float initBalance, String bankID, Random rand, int nStepsBalanceHistory) {
		this.id = id;
		this.setBalance(initBalance);
		this.bankID = bankID;
		this.random = rand;

		this.cashInModel = new CashInModel();
		this.cashInModel.setAccount(this);
		this.cashInModel.setParameters(interval, -1, -1);

		this.cashOutModel = new CashOutModel();
		this.cashOutModel.setAccount(this);
		this.cashOutModel.setParameters(interval, -1, -1);

		this.accountBehaviour = new AccountBehaviour(this.isSAR);

		// Set monthlyIncome
		this.monthlyIncome = new SalaryDistribution().sample();
		this.monthlyIncomeSar = new SalaryDistribution().sample();

		// Set monthlyOutcome
		this.monthlyOutcome = new TruncatedNormal(0.5 * this.monthlyIncome, 0.1 * this.monthlyIncome,
				0.1 * this.monthlyIncome, 0.9 * this.monthlyIncome).sample();
		this.monthlyOutcomeSar = new TruncatedNormal(0.5 * this.monthlyIncomeSar, 0.1 * this.monthlyIncomeSar,
				0.1 * this.monthlyIncomeSar, 0.9 * this.monthlyIncome).sample();

		// Set balanceHistory
		for (int i = 0; i < nStepsBalanceHistory; i++) {
			this.balanceHistory.add((double) initBalance);
		}
		AMLSim.handleIncome(0, "INITALBALANCE", initBalance, this, false, (long) -1, (long) 11);
	}

	public String getBankID() {
		return this.bankID;
	}

	public long getStartStep() {
		return this.startStep;
	}

	public long getEndStep() {
		return this.endStep;
	}

	void setSAR(boolean flag) {
		this.isSAR = flag;
		this.accountBehaviour.updateParameters(this.isSAR); // if account involved in a single SAR, set its behavior
	}

	public boolean isSAR() {
		return this.isSAR;
	}

	public double getBalance() {
		return this.balance;
	}

	public int getNumberOfPhoneChanges() {
		return this.accountBehaviour.getNumberOfPhoneChanges();
	}

	public void setBalance(double balance) {
		this.balance = balance;
	}

	public boolean withdraw(double amount) {
		// if (this.balance < amount) {
		// this.balance = 0.0;
		// } else {
		// this.balance -= amount;
		// }
		boolean success;
		if (this.balance > amount && amount > 0.0 && this.balance >= 100.0) {
			this.balance -= amount;
			success = true;
		} else {
			success = false;
		}
		return success;
	}

	public void deposit(double amount) {
		this.balance += amount;
	}

	public boolean withdrawCash(double amount) {
		// if (this.cashBalance < ammount) {
		// this.cashBalance = 0;
		// } else {
		// this.cashBalance -= ammount;
		// }
		boolean success;
		if (this.cashBalance > amount && amount > 0.0 && this.cashBalance >= 100.0) {
			this.cashBalance -= amount;
			success = true;
		} else {
			success = false;
		}
		return success;
	}

	public void depositCash(double ammount) {
		this.cashBalance += ammount;
	}

	void setBranch(Branch branch) {
		this.branch = branch;
	}

	public Branch getBranch() {
		return this.branch;
	}

	public void addBeneAcct(Account bene) {
		String beneID = bene.id;
		if (beneAcctIDs.contains(beneID)) { // Already added
			return;
		}

		if (ModelParameters.shouldAddEdge(this, bene)) {
			beneAccts.add(bene);
			beneAcctIDs.add(beneID);

			bene.origAccts.add(this);
			bene.origAcctIDs.add(id);

			if (bene.isSAR) {
				numSARBene++;
			}
		}
	}

	public void addTxType(Account bene, String ttype) {
		this.tx_types.put(bene.id, ttype);
		all_tx_types.add(ttype);
	}

	public String getTxType(Account bene) {
		String destID = bene.id;

		if (this.tx_types.containsKey(destID)) {
			return tx_types.get(destID);
		} else if (!this.tx_types.isEmpty()) {
			List<String> values = new ArrayList<>(this.tx_types.values());
			return values.get(this.random.nextInt(values.size()));
		} else {
			return Account.all_tx_types.get(this.random.nextInt(Account.all_tx_types.size()));
		}
	}

	/**
	 * Get previous (originator) accounts
	 * 
	 * @return Originator account list
	 */
	public List<Account> getOrigList() {
		return this.origAccts;
	}

	/**
	 * Get next (beneficiary) accounts
	 * 
	 * @return Beneficiary account list
	 */
	public List<Account> getBeneList() {
		return this.beneAccts;
	}

	public void printBeneList() {
		System.out.println(this.beneAccts);
	}

	public int getNumSARBene() {
		return this.numSARBene;
	}

	public float getPropSARBene() {
		if (numSARBene == 0) {
			return 0.0F;
		}
		return (float) numSARBene / beneAccts.size();
	}

	/**
	 * Register this account to the specified alert.
	 * 
	 * @param alert Alert
	 */
	public void addAlert(Alert alert) {
		this.alerts.add(alert);
	}

	public void addAccountGroup(AccountGroup accountGroup) {
		this.accountGroups.add(accountGroup);
	}

	public void setProp(double probIncome, double meanIncome, double stdIncome, double probIncomeSar,
			double meanIncomeSar, double stdIncomeSar, double meanOutcome, double stdOutcome, double meanOutcomeSar,
			double stdOutcomeSar, double probSpendCash) {
		this.probIncome = probIncome;
		this.meanIncome = meanIncome;
		this.stdIncome = stdIncome;
		this.probIncomeSar = probIncomeSar;
		this.meanIncomeSar = meanIncomeSar;
		this.stdIncomeSar = stdIncomeSar;
		this.meanOutcome = meanOutcome;
		this.stdOutcome = stdOutcome;
		this.meanOutcomeSar = meanOutcome;
		this.stdOutcomeSar = stdOutcome;
		this.probSpendCash = probSpendCash;
	}

	/**
	 * Perform transactions
	 * 
	 * @param state AMLSim object
	 */
	@Override
	public void step(SimState state) {
		long currentStep = state.schedule.getSteps(); // Current simulation step
		long start = this.startStep >= 0 ? this.startStep : 0;
		long end = this.endStep > 0 ? this.endStep : AMLSim.getNumOfSteps();
		this.balanceHistory.removeFirst();
		this.balanceHistory.addLast(this.balance);

		if (!this.isSAR) {
			// Handle salary, if 25th of the month, deposit salary
			if (currentStep % 28 == 25) {
				AMLSim.handleIncome(currentStep, "TRANSFER", this.monthlyIncome, this, this.isSAR, (long) -1,
						(long) 11);
			}
			// Handle income
			if (this.random.nextDouble() < this.probIncome) {
				TruncatedNormal tn = new TruncatedNormal(this.meanIncome, this.stdIncome, 0, 1000000);
				double amt = tn.sample();
				AMLSim.handleIncome(currentStep, "TRANSFER", amt, this, this.isSAR, (long) -1, (long) 0);
			}
			// Handle monthly outcome, if 26th to 28th of the month, pay monthly expense
			if (currentStep == this.stepMonthlyOutcome) {
				AMLSim.handleOutcome(currentStep, "TRANSFER", this.monthlyOutcome, this, this.isSAR, (long) -1,
						(long) 11);
				int diff = this.stepMonthlyOutcome % 28 - 25;
				diff = diff < 0 ? 3 : diff;
				this.stepMonthlyOutcome = this.stepMonthlyOutcome + 28 - diff + random.nextInt(4);
			}
			// Handle outcome
			double meanBalance = 0.0;
			for (double balance : balanceHistory) {
				meanBalance += balance / 28;
			}
			// System.out.println("meanBalance = " + meanBalance + ", balance = " +
			// this.balance + ", inital balance = " + balanceHistory[0]);
			double x = (this.balance - meanBalance) / meanBalance;
			double sigmoid = 1 / (1 + Math.exp(-x));
			if (this.random.nextDouble() < sigmoid) {
				TruncatedNormal tn = new TruncatedNormal(this.meanOutcome, this.stdOutcome, 0.0, 0.9 * this.balance);
				double amt = tn.sample();
				if (this.balance > amt && amt > 0.0 && this.balance >= 100.0) {
					AMLSim.handleOutcome(currentStep, "TRANSFER", amt, this, this.isSAR, (long) -1, (long) 0);
				}
			}
		} else {
			// Handle salary, if 25th of the month, deposit salary
			if (currentStep % 28 == 25) {
				AMLSim.handleIncome(currentStep, "TRANSFER", this.monthlyIncomeSar, this, this.isSAR, (long) -1,
						(long) 11);
			}
			// Handle income
			if (this.random.nextDouble() < this.probIncomeSar) {
				TruncatedNormal tn = new TruncatedNormal(this.meanIncomeSar, this.stdIncomeSar, 1.0, 1000000); // TODO:
																												// handle
																												// lb
																												// better,
																												// maybe
																												// define
																												// in
																												// conf.json?
				double amt = tn.sample();
				AMLSim.handleIncome(currentStep, "TRANSFER", amt, this, this.isSAR, (long) -1, (long) 0);
			}
			// Handle monthly outcome, if 26th to 28th of the month, pay monthly expense
			if (currentStep == this.stepMonthlyOutcome) {
				AMLSim.handleOutcome(currentStep, "TRANSFER", this.monthlyOutcomeSar, this, this.isSAR, (long) -1,
						(long) 11);
				int diff = (this.stepMonthlyOutcome % 28) - 25;
				diff = diff < 0 ? 3 : diff;
				int nextStep = this.stepMonthlyOutcome + 28 - diff + random.nextInt(4);
				this.stepMonthlyOutcome = nextStep;
			}
			// Handle outcome
			double meanBalance = 0.0;
			for (double balance : balanceHistory) {
				meanBalance += balance / 28;
			}
			meanBalance = meanBalance <= 100.0 ? 1000.0 : meanBalance;
			double x = (this.balance + cashBalance - meanBalance) / meanBalance;
			double sigmoid = 1 / (1 + Math.exp(-x));
			if (this.random.nextDouble() < sigmoid) {
				TruncatedNormal tn = new TruncatedNormal(this.meanOutcomeSar, this.stdOutcomeSar, 0.0,
						0.9 * this.balance); // TODO: handle lb better, maybe define in conf.json?
				double amt = tn.sample();
				if (this.cashBalance > amt && amt > 0.0 && this.random.nextDouble() < this.probSpendCash) {
					AMLSim.handleOutcome(currentStep, "CASH", amt, this, this.isSAR, (long) -1, (long) 0);
				} else {
					if (this.balance > amt && amt > 0.0 && this.balance >= 100.0) {
						AMLSim.handleOutcome(currentStep, "TRANSFER", amt, this, this.isSAR, (long) -1, (long) 0);
					}
				}
			}
		}

		this.bankID = this.accountBehaviour.getNewBank(this.bankID);
		this.accountBehaviour.update();
		if (currentStep < start || end < currentStep) {
			return; // Skip transactions if this account is not active
		}
		handleAction(state);
	}

	public void handleAction(SimState state) {
		AMLSim amlsim = (AMLSim) state;
		long step = state.schedule.getSteps();

		for (Alert alert : this.alerts) { // go through the alert patterns the account is involved in
			if (this == alert.getMainAccount()) { // Check if account is the main account of the alert
				alert.registerTransactions(step, this);
			}
		}

		for (AccountGroup accountGroup : this.accountGroups) {
			Account account = accountGroup.getMainAccount();
			if (this == accountGroup.getMainAccount()) {
				accountGroup.registerTransactions(step, account);
			}

		}

		handleCashTransaction(amlsim);
	}

	/**
	 * Make cash transactions (deposit and withdrawal )
	 */
	private void handleCashTransaction(AMLSim amlsim) {
		long step = amlsim.schedule.getSteps();
		this.cashInModel.makeTransaction(step);
		this.cashOutModel.makeTransaction(step);
	}

	public String getName() {
		return this.id;
	}

	public String getID() {
		return this.id;
	}

	public long getDaysInBank() {
		return this.accountBehaviour.getDaysInBank();
	}

	/*
	 * 
	 * Get the account identifier as String
	 * 
	 * @return Account identifier
	 */
	public String toString() {
		return "C" + this.id;
	}

	public ArrayList<String> getParamFile() {
		return paramFile;
	}

	public void setParamFile(ArrayList<String> paramFile) {
		this.paramFile = paramFile;
	}
}
