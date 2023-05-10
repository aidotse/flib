package amlsim.dists;

/**
 * Truncated normal class that constructs and samples faster,
 * at the loss of accuracy.
 * 
 * TODO implement other methods.
 * 
 * @author mao
 *
 */
public class TruncatedNormalQuick extends TruncatedNormal {
	
	private static final long serialVersionUID = 2156605237932737423L;
		
	public TruncatedNormalQuick(double mean, double sd, double lb, double ub) {
		super(mean, sd, lb, ub, 
				NormalDistQuick.cdf01((lb - mean)/sd), 
				NormalDistQuick.cdf01((ub - mean)/sd));		
	}
		
	@Override
	public double density(double x) {
		if( x <= aa || x >= bb ) return 0;
		
		return NormalDistQuick.density01((x - mu)/sigma) / (sigma * Z);
	}

	@Override
	public double sample() {
		double val = super.random.nextDouble() * Z + cdf_a;
		
		return mu + sigma * NormalDistQuick.inverseF01(val);
	}

}