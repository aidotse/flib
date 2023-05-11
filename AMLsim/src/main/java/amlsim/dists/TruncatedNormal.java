package amlsim.dists;
import org.apache.commons.math3.distribution.AbstractRealDistribution;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

/**
 * Truncated normal distribution, mostly from
 * http://en.wikipedia.org/wiki/Truncated_normal_distribution
 * However, that page has some errors, which have been corrected.
 * 
 * @author mao
 *
 */
public class TruncatedNormal extends AbstractRealDistribution {

	private static final long serialVersionUID = 3279850685379757823L;

	protected final double mu;
	protected final double sigma;	
	protected final double cdf_a;
	protected final double cdf_b;
	protected final double Z;
	
	protected final double alpha;
	protected final double beta;
	protected final double aa;
	protected final double bb;
	
	public TruncatedNormal(RandomGenerator rng, double mean, double sd, double lb, double ub) {
		super(rng);
		
		if( mean == Double.NaN || sd == Double.NaN ||
				lb == Double.NaN || ub == Double.NaN ) 
			throw new IllegalArgumentException("Cannot take NaN as an argument");				
		
		mu = mean;
		sigma = sd;
		cdf_a = NormalDist.cdf01((lb - mu)/sigma);
		cdf_b = NormalDist.cdf01((ub - mu)/sigma);
		Z = cdf_b - cdf_a;
		
		alpha = (lb - mu) / sigma;
		beta = (ub - mu) / sigma;
		this.aa = lb;
		this.bb = ub;
	}
	
	protected TruncatedNormal(double mean, double sd, double lb, double ub, double cdf_a, double cdf_b) {
		super(new Well19937c());
		
		this.mu = mean;
		this.sigma = sd;
		this.cdf_a = cdf_a;
		this.cdf_b = cdf_b;
		this.Z = cdf_b - cdf_a;
		
		this.alpha = (lb - mu) / sigma;
		this.beta = (ub - mu) / sigma;
		this.aa = lb;
		this.bb = ub;
	}
	
	public TruncatedNormal(double mean, double sd, double lb, double ub) { 
		this(new Well19937c(), mean, sd, lb, ub);
	}
	
	@Override
	public double probability(double x) {		
		return 0;
	}

	@Override
	public double density(double x) {
		if( x <= aa || x >= bb ) return 0;
		
		return NormalDist.density01((x - mu)/sigma) / (sigma * Z);
	}

	@Override
	public double cumulativeProbability(double x) {
		if( x <= aa ) return 0;
		else if( x >= bb ) return 1;
			
		double u = NormalDist.cdf01((x - mu)/sigma);
		return (u - cdf_a) / Z;
	}

	@Override
	public double inverseCumulativeProbability(double p) throws OutOfRangeException {		
		if (p < 0.0 || p > 1.0) throw new OutOfRangeException(p, 0, 1);        
		
		double val = p * Z + cdf_a;
		
		return mu + sigma * NormalDist.inverseF01(val);
	}

	@Override
	public double getNumericalMean() {
		double phi_a = NormalDist.density01(alpha);
		double phi_b = NormalDist.density01(beta);
		
		return mu + (phi_a - phi_b) * sigma / Z;
	}

	@Override
	public double getNumericalVariance() {
		double phi_a = NormalDist.density01(alpha);
		double phi_b = NormalDist.density01(beta);
		double sq = ( phi_a - phi_b ) / Z;
		double br = 1 + ( alpha * phi_a - beta * phi_b ) / Z - sq * sq;  
		return sigma * sigma * br;
	}

	@Override
	public double sample() {			
		double val = super.random.nextDouble() * Z + cdf_a;
		
		return mu + sigma * NormalDist.inverseF01(val);
	}

	@Override
	public double getSupportLowerBound() {		
		return aa;
	}

	@Override
	public double getSupportUpperBound() {		
		return bb;
	}

	@Override
	public boolean isSupportLowerBoundInclusive() {		
		return false;
	}

	@Override
	public boolean isSupportUpperBoundInclusive() {		
		return false;
	}

	@Override
	public boolean isSupportConnected() {		
		return true;
	}

}