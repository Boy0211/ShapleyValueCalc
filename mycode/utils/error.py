import math
import pandas as pd
import os

from tqdm import trange

class Errors():

    def __init__(self, path) -> None:
        self.path = path + '/data'

    def run(self, error_type, save=False):

        # Check if the errors are already calculated
        if self.error_check(error_type) is not None:
            return self.error_check(error_type)

        # List all data in experiment path
        files = os.listdir(self.path)

        errors = []
        for indexer in trange(len(files)):
            
            # Load the file in
            file = files[indexer]
            with open(f'{self.path}/{file}', 'r') as f:
                lines = f.readlines()

            # Load in the metadata
            metadata = {}
            for line in lines:
                
                # if the row contains metadata
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                else:
                    insert_index = lines.index(line)
                    break

            # Load in the data
            data = pd.read_csv(f'{self.path}/{file}', 
                               skiprows=insert_index+2, 
                               names=['Player', 'ShapleyValue'])

            # if the error already exists in the metadata, skip
            if error_type not in metadata:
                
                # Calculate the error
                metadata[error_type] = self.compare(metadata['Number of Players'], 
                                                    metadata['Game'], 
                                                    data['ShapleyValue'], 
                                                    error_type)

                # Insert the error into the file
                with open(f'{self.path}/{file}', 'r') as f:
                    lines = f.readlines()

                # If the error has not been previously calculated
                lines.insert(insert_index, f'{error_type}: {metadata[error_type]}\n')
                with open(f'{self.path}/{file}', 'w') as f:
                    f.writelines(lines)

            # Append the metadata to the errors list
            errors.append(metadata)

        # Create a dataframe from the errors list
        df = pd.DataFrame(errors)

        # Save the dataframe if save is True
        if save:
            
            # Check if the errors are already calculated
            if self.error_check(error_type) is None:      
                df.to_csv(self.path.split('/data')[0] + '/errors.csv', index=False)

        return df
        
    def compare(self, n, game_name, output, error_type):

        if game_name.split('_')[0] == 'networkedgame':
            
            # read in the real values
            graphtype = game_name.split('_')[1]
            graphtype = graphtype.lower()
            gametype = game_name.split('_')[2]
            real_values = pd.read_csv(f'data/messagegame/{gametype}/{graphtype}/{str(n)}.csv')
            real_values = real_values['ShapleyValue'].to_dict()

        else:
            # read in the real values
            real_values = pd.read_csv(f'data/{game_name}/{str(n)}.csv')
            real_values = real_values['ShapleyValue'].to_dict()

        # calculate the errors
        error_type = getattr(self, error_type)
        error = error_type(real_values, output)

        return error
    
    def error_check(self, error_type):
        
        # Check if the errors are already calculated
        if os.path.exists(self.path.split('/data')[0] + '/errors.csv'):
            
            # Check if length of the df is equal to the number of files
            df = pd.read_csv(self.path.split('/data')[0] + '/errors.csv')
            files = os.listdir(self.path)

            if len(df) != len(files):
                return None
            
            # Check if the error type is in the dataframe
            if error_type in df.columns:

                # check if there are NaN in the column presented
                if df[error_type].isnull().values.any():
                    return None
                else:
                    return df
            else:
                return None
        else:
            return None
    
    def SAE(self, sh_real, sh_est):
        """Sum of the absolute error"""
        error = 0
        for i in range(len(sh_est)):
            error += abs(sh_est[i] - sh_real[i])

        return error
    
    def MeanAbsoluteError(self, sh_real, sh_est):
        """Mean absolute error"""
        # Calculate the Mean Absolute Error (MAE) between two lists
        error = 0
        for i in range(len(sh_est)):
            error += abs(sh_est[i] - sh_real[i])
        return error/len(sh_est)
    
    def AARE(self, sh_real, sh_est):
        """Average absolute error ratio"""

        error = 0
        for i in range(len(sh_est)):
            error += abs((sh_est[i] - sh_real[i]) / sh_real[i])
        return error/len(sh_est)
    
    def MAE(self, sh_real, sh_est):
        """Max absolute error"""

        # Calculate the Max Absolute Error (MAE) between two lists
        error = 0
        for i in range(len(sh_est)):
            error = max(error, abs(sh_est[i] - sh_real[i]))
        return error
    
    def MSE(self, sh_real, sh_est):
        """Mean squared error"""
        # Calculate the Mean Squared Error (MSE) between two lists
        error = 0
        for i in range(len(sh_est)):
            error += (sh_est[i] - sh_real[i])**2

        return error/len(sh_est)

    def MaxErrorRatio(self, sh_real, sh_est):
        """Maximum error ratio"""
        # Calculate the Maximum Error Ratio (MER) between two lists
        error = 0
        for i in range(len(sh_est)):
            error = max(error, (abs(sh_est[i] - sh_real[i]) / sh_real[i]))
        return error

    def RMSE(self, sh_real, sh_est):
        """Root Mean Squared Error"""
        
        # Calculate the Root Mean Squared Error (RMSE) between two lists
        return math.sqrt(self.MSE(sh_real, sh_est))
    
    def Variance(self, sh_real, sh_est):
        """Variance"""
        # Calculate the variance between two lists
        variance = 0
        for i in range(len(sh_est)):
            variance += (sh_est[i] - sh_real[i])**2

        return variance
    
    def MinAE(self, sh_real, sh_est):
        """Minimum absolute error"""
        # Calculate the Minimum Absolute Error (MinAE) between two lists
        error = 0
        for i in range(len(sh_est)):
            error = min(error, abs(sh_est[i] - sh_real[i]))
        return error