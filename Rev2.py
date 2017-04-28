import numpy as np
from graphlab import SFrame, cross_validation
import pickle as pkl
import time

class node(object):
    def __init__(self, isLeaf, splitting_feature, split_value, data,
                 leftChild = None, rightChild = None):
        self.isLeaf = isLeaf
        self.splitting_feature = splitting_feature
        self.split_value = split_value
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.data = data
    
    def gain(self):
        data_left = self.data[self.data[self.splitting_feature]\
                                < self.split_value]
        data_right = self.data[self.data[self.splitting_feature]\
                                >= self.split_value]
        entropy_left_split = (len(data_left) / float(len(self.data)))\
                                * entropy(data_left)
        entropy_right_split = (len(data_right) / float(len(self.data)))\
                                * entropy(data_right)
        return entropy(self.data) - entropy_left_split - entropy_right_split
    
    def split_info(self):
        data_left = len(self.data[self.data[self.splitting_feature]\
                                < self.split_value])
        data_right = len(self.data[self.data[self.splitting_feature]\
                                >= self.split_value])
        ratio_left = data_left / float(len(self.data))
        ratio_right = data_right / float(len(self.data))
        log_left = np.log2(ratio_left)
        log_right = np.log2(ratio_right)
        if log_left == -np.inf:
            log_left = 0.0
        if log_right == -np.inf:
            log_right == 0.0
        return - (ratio_left * log_left + ratio_right * log_right)
    
    def gain_ratio(self):
        return self.gain() / self.split_info()
    
    def prediction(self):
        if (self.data['diagnosis'] == 'M').sum() >\
            (self.data['diagnosis'] == 'B').sum():
            return 'M'
        else:
            return 'B'

def entropy(data):
    '''
    Calculate Entropy of the given data
    '''
    if len(data) != 0:
        total_data = len(data)
        m_freq = (data['diagnosis'] == 'M').sum() / float(total_data)
        b_freq = (data['diagnosis'] == 'B').sum() / float(total_data)
        if m_freq == 0:
            log2_m = 0.0
        else:
            log2_m = np.log2(m_freq)
        if b_freq == 0:
            log2_b = 0.0
        else:
            log2_b = np.log2(b_freq)
        return -(m_freq * log2_m + b_freq * log2_b)
    else :
        return 0

def best_splitting_feature_gain(data, features, isContinuous=True,
                                annotate=False):
    '''
    Calculate best splitting feature through Entropy
    '''
    n = float((data['diagnosis'] == 'M').sum())
    p = float((data['diagnosis'] == 'B').sum())
    entropy_data = entropy(data)
    if annotate:
        print "Entropy of data: %f" % entropy_data

    best_feature = None
    best_gain = -np.inf

    for feature in features:
        if annotate:
            print 'Working on ' + feature
        if isContinuous:
            data_sorted = data[feature].sort()
            data_points = [np.mean([data_sorted[i], data_sorted[i + 1]]) \
                            for i in range(len(data_sorted) - 1)]
            gain = []
            for i in data_points:
                #For points <= split
                pi_tmp = float(((data['diagnosis'] == 'B') & (data[feature] <= i)).sum())
                ni_tmp = float(((data['diagnosis'] == 'M') & (data[feature] <= i)).sum())
                left_split = float((pi_tmp + ni_tmp) / (p + n))
                log_left = np.log2(left_split)
                if log_left == -np.inf:
                    log_left = 0.0
                left_split_info = float(left_split * log_left)
                gain.append(((pi_tmp + ni_tmp) / (p + n)) * entropy(data[data[feature] <= i]))

                #For points > split
                pi_tmp = float(((data['diagnosis'] == 'B') & (data[feature] > i)).sum())
                ni_tmp = float(((data['diagnosis'] == 'M') & (data[feature] > i)).sum())
                right_split = float((pi_tmp + ni_tmp) / (p + n))
                log_right = np.log2(right_split)
                if log_right == -np.inf:
                    log_right = 0.0
                right_split_info = float(right_split * log_right)
                gain.append(((pi_tmp + ni_tmp) / (p + n)) * entropy(data[data[feature] > i]))

            gain_split = entropy_data - float(sum(gain))
            gain_ratio = gain_split / (-(left_split_info + right_split_info))
            if annotate:
                print 'Gain Ratio from %s: %f' % (feature, gain_ratio)
            if gain_ratio > best_gain:
                best_gain = gain_ratio
                best_feature = feature
        else:
            print 'TODO: Implement discrete valued Information Gain'

    return best_feature, np.median(data[best_feature])

def intermediate_node_num_mistakes(labels_in_node):
    '''
    Counts basic error in current split based on minority class
    '''
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0

    # Count the number of B's (benign tumors)
    num_b = (labels_in_node == 'B').sum()

    # Count the number of M's (malignant tumors)
    num_m = (labels_in_node == 'M').sum()

    # Return the number of mistakes that the majority classifier makes.
    return min(num_b, num_m)

def create_pre_pruned(data, features, current_depth=0, max_depth=10,
                        threshold=10):
    '''
    Create a decision tree with pre-pruning similar to ID3 or C4.5
    '''
    target_values = data['diagnosis']
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    # Stopping Condition 1
    # Stop at pure nodes
    if intermediate_node_num_mistakes(data['diagnosis']) == 0:
        print 'Pure node reached'
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = data)
    
    # Stopping Condition 2
    # Stop when number of datapoints fall below a threshold
    if len(data) <= threshold:
        print 'Less than %d datapoints left' % threshold
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = data)
    
    # Stopping Condition 3
    # Stop after specified tree depth
    if current_depth >= max_depth:
        print 'Max depth reached (%d)' % max_depth
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = data)
    
    # Finding best split through gain ratio
    splitting_feature, split_value = best_splitting_feature_gain(data, features, 'diagnosis')
    left_split = data[data[splitting_feature] < split_value]
    right_split = data[data[splitting_feature] >= split_value]
    print 'Split on feature %s at %f' % (splitting_feature, split_value)

    # Create a leaf node if split is perfect
    if len(right_split) == 0:
        print 'Creating leaf node for left data'
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = left_split)
    if len(left_split) == 0:
        print 'Creating leaf node for right data'
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = right_split)
    
    # Recurse on left and right subtrees
    left_tree = create_pre_pruned(left_split, features, current_depth+1, max_depth)
    right_tree = create_pre_pruned(right_split, features, current_depth+1, max_depth)

    return node (
        isLeaf = False,
        splitting_feature = splitting_feature,
        split_value = split_value,
        leftChild = left_tree,
        rightChild = right_tree,
        data = data
    )

def count_nodes(tree):
    '''
    Return number of nodes in the tree
    '''
    if tree.isLeaf:
        return 1
    return 1 + count_nodes(tree.leftChild) + count_nodes(tree.rightChild)

def classify(tree, x, annotate=False):
    '''
    Returns prediction for a row
    '''
    # if the node is a leaf node.
    if tree.isLeaf:
        if annotate:
            print "At leaf, predicting %s" % tree.prediction()
        return tree.prediction()
    else:
        # split on feature.
        feature_value = x[tree.splitting_feature]
        split_value = tree.split_value
        if annotate:
            print "Split on %s = %s" % (tree.splitting_feature, feature_value)
        if feature_value < split_value:
            return classify(tree.leftChild, x, annotate)
        else:
            return classify(tree.rightChild, x, annotate)

def evaluate_classification_error(tree, data, annotate=False):
    '''
    Returns classification error
    '''
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x, annotate=annotate))

    # Once you've made the predictions, calculate the classification error and return it
    return (data['diagnosis'] != prediction).sum() / float(len(prediction))

def create_unbounded(data, features, current_depth=0):
    '''
    Create a decison tree without any checks.
    It is inadvisable to use this without prune().
    '''
    target_values = data['diagnosis']
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stop at pure nodes
    if intermediate_node_num_mistakes(data['diagnosis']) == 0:
        print 'Pure node reached'
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = data)
    
    # Finding best split through gain ratio
    splitting_feature, split_value = best_splitting_feature_gain(data, features, 'diagnosis')
    left_split = data[data[splitting_feature] < split_value]
    right_split = data[data[splitting_feature] >= split_value]
    print 'Split on feature %s at %f' % (splitting_feature, split_value)

    # Create a leaf node if split is perfect
    if len(left_split) == len(data):
        print 'Creating leaf node'
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = left_split)
    if len(right_split) == len(data):
        print 'Creating leaf node'
        return node(isLeaf = True, splitting_feature = None,
                        split_value = None, data = right_split)
    
    # Recurse on left and right subtrees
    left_tree = create_unbounded(left_split, features, current_depth+1)
    right_tree = create_unbounded(right_split, features, current_depth+1)

    return node (
        isLeaf = False,
        splitting_feature = splitting_feature,
        split_value = split_value,
        leftChild = left_tree,
        rightChild = right_tree,
        data = data
    )

def prune(tree, test_data):
    '''
    Prunes a pre-built tree. Returns pruned tree and least error encountered.
    '''
    bfs = [tree]
    bfs_iter = 0

    while(True):
        if bfs_iter == len(bfs):
            break
        if bfs[bfs_iter].leftChild != None:
            bfs.append(bfs[bfs_iter].leftChild)
        if bfs[bfs_iter].rightChild != None:
            bfs.append(bfs[bfs_iter].rightChild)
        bfs_iter += 1
    
    least_error = evaluate_classification_error(tree, test_data)
    snips = 0

    bfs.reverse()
    for i in bfs:
        i.isLeaf = True
        error_partial = evaluate_classification_error(tree, test_data)
        if error_partial <= least_error:
            least_error = error_partial
            snips += 1
        else:
            i.isLeaf = False
    print 'Made %d snips' % snips
    return tree, least_error

if __name__ == '__main__':
    data = SFrame.read_csv('data.csv', column_type_hints = [str, str] + [float]*30)
    benign_raw, malign_raw = data[data['diagnosis'] == 'B'], data[data['diagnosis'] == 'M']
    percentage = len(malign_raw) / float(len(benign_raw))
    benign = benign_raw.sample(percentage)
    malign = malign_raw
    data = benign.append(malign)
    data = cross_validation.shuffle(data)

    print "Percentage of benign tumors              :", len(benign) / float(len(data))
    print "Percentage of malignant tumors           :", len(malign) / float(len(data))
    print "Total number of tumors in new dataset    :", len(data)
    train_data, test_data = data.random_split(.8)
    target = 'diagnosis'
    features = data.column_names()[2:]
    
    # Test function. SHould return perimeter_worst and its median
    # print best_splitting_feature_gain(data, data.column_names()[2:], 'diagnosis', annotate=True)

    # Create a pre-pruned tree and save that model for later use
    time_pre = time.time()
    model_pre_pruned = create_pre_pruned(data, features)
    print 'Pre-pruned model took {} seconds'.format(time.time() - time_pre)
    with open('model_pre_pruned.pkl', 'wb') as output:
        pkl.dump(model_pre_pruned, output, -1)
    
    # del model_pre_pruned
    try:
        with open('model_pre_pruned.pkl', 'rb') as inp:
            model_pre_pruned = pkl.load(inp)
    except Exception:
        print 'Cannot read pre-pruned model binary file'
    
    # Create an unbounded tree for post-pruning later. Save the model
    time_unbounded = time.time()
    model_unbounded = create_unbounded(data, features)
    print 'Unbounded model took {} seconds'.format(time.time() - time_unbounded)
    with open('model_unbounded.pkl', 'wb') as output:
        pkl.dump(model_unbounded, output, -1)
    
    # del model_unbounded
    try:
        with open('model_unbounded.pkl', 'rb') as inp:
            model_unbounded = pkl.load(inp)
    except Exception:
        print 'Cannot read unbounded model binary file'

    # Now prune the unbounded tree and save this too
    time_post_pruned = time.time()
    model_post_pruned = prune(model_unbounded, test_data)
    print 'Pruning model took {} seconds'.format(time.time() - time_post_pruned)
    with open('model_post_pruned.pkl', 'wb') as output:
        pkl.dump(model_post_pruned, output, -1)
    
    # del model_post_pruned
    try:
        with open('model_post_pruned.pkl', 'rb') as inp:
            model_post_pruned = pkl.load(inp)
    except Exception:
        print 'Cannot read post pruned model binary file'
    
    # Let's try classifying some rows from the test_data
    # Let's look at the first row in the test_data
    print '----------------Test Data Row----------------------'
    print test_data[0]
    print '---------------------------------------------------'
    # What does the post_pruned model say?
    print classify(model_post_pruned, test_data[0], annotate=True)
    # What does the pre_pruned model say?
    print classify(model_pre_pruned, test_data[0], annotate=True)

    # What is the overall classification error?
    print 'Pre-pruned: %f' % evaluate_classification_error(model_pre_pruned, data)
    print 'Post-pruned: %f' % evaluate_classification_error(model_post_pruned, data)
