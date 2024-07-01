package amlsim.dists;

/*
 * Originally from 
 * http://www.iro.umontreal.ca/~simardr/ssj/indexe.html
 */

 public class Num {

	/**
	 * The value of <SPAN CLASS="MATH">(2)<SUP>1/2</SUP></SPAN>.
	 * 
	 */
	public static final double RAC2 = 1.41421356237309504880;

	/**
	 * Contains the precomputed positive powers of 2.
	 *    One has <TT>TWOEXP[j]</TT><SPAN CLASS="MATH">= 2<SUP>j</SUP></SPAN>, for 
	 * <SPAN CLASS="MATH"><I>j</I> = 0,..., 64</SPAN>.
	 * 
	 */
	public static final double TWOEXP[] = {
		1.0, 2.0, 4.0, 8.0, 1.6e1, 3.2e1,
		6.4e1, 1.28e2, 2.56e2, 5.12e2, 1.024e3,
		2.048e3, 4.096e3, 8.192e3, 1.6384e4, 3.2768e4,
		6.5536e4, 1.31072e5, 2.62144e5, 5.24288e5,
		1.048576e6, 2.097152e6, 4.194304e6, 8.388608e6,
		1.6777216e7, 3.3554432e7, 6.7108864e7,
		1.34217728e8, 2.68435456e8, 5.36870912e8,
		1.073741824e9, 2.147483648e9, 4.294967296e9,
		8.589934592e9, 1.7179869184e10, 3.4359738368e10,
		6.8719476736e10, 1.37438953472e11, 2.74877906944e11,
		5.49755813888e11, 1.099511627776e12, 2.199023255552e12,
		4.398046511104e12, 8.796093022208e12,
		1.7592186044416e13, 3.5184372088832e13,
		7.0368744177664e13, 1.40737488355328e14,
		2.81474976710656e14, 5.62949953421312e14,
		1.125899906842624e15, 2.251799813685248e15,
		4.503599627370496e15, 9.007199254740992e15,
		1.8014398509481984e16, 3.6028797018963968e16,
		7.2057594037927936e16, 1.44115188075855872e17,
		2.88230376151711744e17, 5.76460752303423488e17,
		1.152921504606846976e18, 2.305843009213693952e18,
		4.611686018427387904e18, 9.223372036854775808e18,
		1.8446744073709551616e19
	};

	/**
	 * Evaluates a series of Chebyshev polynomials <SPAN CLASS="MATH"><I>T</I><SUB>j</SUB></SPAN> at
	 *   <SPAN CLASS="MATH"><I>x</I></SPAN> over the basic interval <SPAN CLASS="MATH">[- 1, &nbsp;1]</SPAN>. It uses
	 *    the method of Clenshaw, i.e., computes and  returns
	 *   
	 * <P></P>
	 * <DIV ALIGN="CENTER" CLASS="mathdisplay">
	 * <I>y</I> = <IMG
	 *  ALIGN="MIDDLE" BORDER="0" SRC="Numimg4.png"
	 *  ALT="$\displaystyle {\frac{{a_0}}{2}}$"> + &sum;<SUB>j=1</SUB><SUP>n</SUP><I>a</I><SUB>j</SUB><I>T</I><SUB>j</SUB>(<I>x</I>).
	 * </DIV><P></P>
	 * @param a coefficients of the polynomials
	 * 
	 *        @param n largest degree of polynomials
	 * 
	 *        @param x the parameter of the <SPAN CLASS="MATH"><I>T</I><SUB>j</SUB></SPAN> functions
	 * 
	 *        @return  the value of a series of Chebyshev polynomials <SPAN CLASS="MATH"><I>T</I><SUB>j</SUB></SPAN>.
	 * 
	 */
	public static double evalCheby (double a[], int n, double x)  {
		if (Math.abs (x) > 1.0)
			System.err.println ("Chebychev polynomial evaluated "+
					"at x outside [-1, 1]");
		final double xx = 2.0*x;
		double b0 = 0.0;
		double b1 = 0.0;
		double b2 = 0.0;
		for (int j = n; j >= 0; j--) {
			b2 = b1;
			b1 = b0;
			b0 = (xx*b1 - b2) + a[j];
		}
		return (b0 - b2)/2.0;
	}
}