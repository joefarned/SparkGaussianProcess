import breeze.linalg.DenseMatrix
import breeze.linalg.cholesky
import breeze.linalg.Axis.{_0, _1}
import breeze.linalg.DenseVector
import breeze.linalg.*
import breeze.stats.distributions.Gaussian
import breeze.math._
import breeze.numerics._
import breeze.optimize._
import breeze.math.Complex
import breeze.optimize.ApproximateGradientFunction
import breeze.optimize.StochasticDiffFunction

object GaussianProcess {
  var L = new DenseMatrix[Double](0, 0)
  var Xtrain = new DenseMatrix[Double](0, 0)
  var Ytrain = new DenseVector[Double](0)
  var best = 1E200 * 1E200 // Arbitrarily large number

  /**
    Distance between two vector
  **/
  def distance(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val square = (x - y).t * (x - y)
    sqrt(square)
  }

  /**
    Squared exponential kernel
  **/
  def covariance(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val pow = -0.5 * distance(x, y)
    exp(pow)
  }

  /**
    Kernel between two dense matrices
  **/
  def kernel(x1: DenseMatrix[Double], x2:DenseMatrix[Double]): DenseMatrix[Double] = {
    val K = DenseMatrix.zeros[Double](x1.cols, x2.cols)

    for(i <- 0 to x1.cols - 1) {
      for(j <- 0 to x2.cols - 1) {
        K(i, j) = covariance(x1(::, i), x2(::, j))
      }
    }
    K
  }

  /**
    Calculate covariance matrix for training data, plus its Cholesky decomposition
  **/
  def train(xtrain: DenseMatrix[Double], ytrain: DenseVector[Double]) = {
    Xtrain = xtrain
    Ytrain = ytrain

    val K = kernel(Xtrain, Xtrain)
    L = cholesky(K)
  }

  /**
    GP regression, given test and training data
  **/
  def predict(x: DenseVector[Double]): (Double, Double) = {

    val Xtest = x.toDenseMatrix.t
    val K_s = kernel(Xtrain, Xtest)
    val v = L \ K_s

    val mu = v.t * (L \ Ytrain)
    val cov = kernel(Xtest, Xtest) - v.t * v

    (mu(0), cov(0, 0))
  }

  /**
    Expected improvement metric for Bayesian Optimization
    Compare with probability of improvement
  **/
  def expectedImprovement(x: DenseVector[Double]): Double = {
    best = breeze.linalg.min(Ytrain.toDenseMatrix)

    val (mu, sigma) = predict(x)

    val g = Gaussian(0, 1)
    val Z = (best - mu) / (sigma + 0.000001)
    g.pdf(Z) + (sigma * Z * g.cdf(Z))
  }

  /**
    Function to be minimized for Bayesian Optimization
  **/
  val objective = new DiffFunction[DenseVector[Double]] {
    def calculate(x: DenseVector[Double]) = {
      val diffg = new ApproximateGradientFunction(expectedImprovement)
      (-expectedImprovement(x), diffg.gradientAt(x).toDenseVector)
    }
  }
}
