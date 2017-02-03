#' @title Variable Importance
#'
#' @description
#' With the help of this function the permutation variable importance for random forests 
#' can be created for any measure that is available in the mlr package.
#'
#' @param mod
#'   An object of class randomForest, as that created by the function randomForest with option keep.inbag = TRUE
#' @param measures
#'   List of performance measure(s) of mlr to evaluate. Default is auc only.
#' @param task
#'   Learning task created by the function makeClassifTask or makeRegrTask of mlr. 
#' @return
#'   Returns a dataframe with a column for each desired measure.
#' @export
#' @examples
#' library(mlr)
#' library(randomForest)
#' 
#' # Classification
#' data = getTaskData(iris.task)
#' mod = randomForest(Species ~., data = data, ntree = 100, keep.inbag = TRUE)
#' results = vimp(mod, measures = list(mmce, multiclass.au1u, multiclass.brier), task = iris.task, data = data)
#' 
vimp = function(mod, measures = list(auc), task, data) {
  tasktype = getTaskType(task)
  target = getTaskTargetNames(task)
  target.column = which(colnames(data)==target)
  truth = mod$y
  preds = predict(mod, newdata = data, predict.all = TRUE)
  inbag = mod$inbag
  nobs = nrow(preds$individual)
  
  # calculate original performance
  oobpreds = predict(mod, type = "prob")
  oob.perf = calculateMlrMeasure(oobpreds, measures, task, truth, predict.type = "prob")
  
  measure.names = BBmisc:::extractSubList(measures, "id")
  oob.diff = array(data = NA, dim = c(ncol(data), length(measures)), dimnames = list(colnames(data), measure.names))
  if (tasktype == "classif") {
    num_levels = nlevels(preds$aggr)
    pred_levels = levels(preds$aggr)
    for(i in 1:ncol(data)){
      data.perm = data[, -target.column]
      data.perm[, i] = data.perm[sample(1:nobs), i] # make it per tree !? Would take too long...
      preds.perm = predict(mod, newdata = data.perm, predict.all = TRUE)$individual
      
      prob_array = array(data = NA, dim = c(nobs, num_levels), dimnames = list(NULL, pred_levels))
      for (j in 1:num_levels){ # This could maybe done faster
        preds.perm.j = (preds.perm == pred_levels[j]) * 1 
        preds.perm.j = preds.perm.j * ((inbag == 0) * 1) # only use observations that are out of bag
        prob_array[,j] = rowSums(preds.perm.j) / rowSums(inbag)
      }
      perm.perf = calculateMlrMeasure(prob_array, measures, task, truth, predict.type = "prob")
      # make it possible to calculate performance for each level!
      oob.diff[i,] = oob.perf - perm.perf
    }
  }
  return(oob.diff)
}

calculateMlrMeasure = function(x, measures, task, truth, predict.type) {
  mlrpred = mlr::makePrediction(task.desc = task$task.desc, row.names = names(truth), id = names(truth), truth = truth,
    predict.type = predict.type, predict.threshold = NULL, y = x, time = NA)
  performance(mlrpred, measures)
}

