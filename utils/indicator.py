import numpy

def Dice(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    # print(numpy.count_nonzero(reference))
    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero((~result) & reference)
    fp = numpy.count_nonzero(result & (~reference))
    # print(tp, fn, fp)
    try:
        dc = (tp + tp) / float(tp + fn + tp + fp)
    except ZeroDivisionError:
        dc = 0.0

    return dc

def Iou(result, reference):

    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero((~result) & reference)
    fp = numpy.count_nonzero(result & (~reference))
    try:
        jc = tp / float(tp + fn + fp)
    except ZeroDivisionError:
        jc = 0.0
    return jc

def Recall(predict, target): #Sensitivity, Recall, true positive rate都一样

    predict = numpy.atleast_1d(predict.astype(numpy.bool_))
    target = numpy.atleast_1d(target.astype(numpy.bool_))

    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero((~predict) & target)


    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall

def Precision(predict, target):

    predict = numpy.atleast_1d(predict.astype(numpy.bool_))
    target = numpy.atleast_1d(target.astype(numpy.bool_))

    tp = numpy.count_nonzero(predict & target)
    fp = numpy.count_nonzero(predict & (~target))


    try:
        ans = tp / float(tp + fp)
    except ZeroDivisionError:
        ans = 0.0

    return ans

def Accuracy(predict, target):

    predict = numpy.atleast_1d(predict.astype(numpy.bool_))
    target = numpy.atleast_1d(target.astype(numpy.bool_))

    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero((~predict) & target)
    fp = numpy.count_nonzero(predict & (~target))
    tn = numpy.count_nonzero((~predict) & (~target))


    try:
        ans = (tp + tn) / float(tp + fp + tn + fn)
    except ZeroDivisionError:
        ans = 0.0

    return ans

def Specificity(predict, target):

    predict = numpy.atleast_1d(predict.astype(numpy.bool_))
    target = numpy.atleast_1d(target.astype(numpy.bool_))

    fp = numpy.count_nonzero(predict & (~target))
    tn = numpy.count_nonzero((~predict) & (~target))


    try:
        ans = tn / float(tn + fp)
    except ZeroDivisionError:
        ans = 0.0

    return ans

def get_ind(predict, target):

    predict = numpy.atleast_1d(predict.astype(numpy.bool_))
    target = numpy.atleast_1d(target.astype(numpy.bool_))
    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero((~predict) & target)
    fp = numpy.count_nonzero(predict & (~target))
    tn = numpy.count_nonzero((~predict) & (~target))

    # print(tp)
    # print(fn)
    # print(fp)
    # print(tn)
    # print(tp+fn+fp+tn)

    try:
        acc = (tp + tn) / float(tp + fp + tn + fn)
    except ZeroDivisionError:
        acc = 0.0

    try:
        spe = tn / float(tn + fp)
    except ZeroDivisionError:
        spe = 0.0

    try:
        pre = tp / float(tp + fp)
    except ZeroDivisionError:
        pre = 0.0

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    try:
        jc = tp / float(tp + fn + fp)
    except ZeroDivisionError:
        jc = 0.0

    try:
        dc = (tp + tp) / float(tp + fn + tp + fp)
    except ZeroDivisionError:
        dc = 0.0

    # print(tp)
    # print(fn)
    # print(fp)
    # print(tn)
    # print(tp+fn+fp+tn, ' #')
    # print(dc, jc, recall, spe, pre, acc)

    return dc, jc, recall, spe, pre, acc





