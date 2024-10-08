from flask import Flask

app = Flask(__name__)
app.secret_key = 'ptx_tools'

#from app import routes_pTx_Tools
#from app import routes_B0_Mapping_Toolbox
from app import routes_B1_Mapping_Toolbox
from app import routes_ROI_Toolbox
from app import routes_B1_Shim_Toolbox
#from app import routes_kT_Points_Toolbox
#from app import routes_2DSelExc_Toolbox
#from app import routes_Spokes_Toolbox