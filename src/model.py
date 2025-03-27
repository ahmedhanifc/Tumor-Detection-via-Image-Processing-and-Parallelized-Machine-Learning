from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, classification_report, recall_score

def splitData(X, y, test_size: float = 0.30):
    '''
        Takes two arguements, X and y, and splits the dataset 
        on the basis of the specified size.
        Returns X_train, X_test, y_train, y_test
    '''
    assert(X.isna().sum().sum() == 0), "***Error:The Dataset has missing values"
    assert(y.isna().sum().sum() == 0), "***Error:The Dataset has missing values"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test
    
def train_classification_model(model, X_train, X_test, y_train, y_test):
    """Function to train a model and print it"""
    try:
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict, zero_division=1)
        recall = recall_score(y_test, y_predict, zero_division = 1)
        f1 = f1_score(y_test, y_predict, zero_division=1)
        labels = [0,1]
        cm = confusion_matrix(y_test, y_predict, labels = labels)
        return (True,model, {"accuracy":accuracy, "precision":precision, "recall":recall,"f1": f1,"cm":cm},)
    except Exception as e:
        #print(f"Error training {model}: {str(e)}")
        return False
    '''

    '''

