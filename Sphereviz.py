# trace generated using paraview version 5.10.0
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10


#### Import the simple module from the paraview
from paraview.simple import *
import os


#### Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


#### Find source to which the subsequent filters will be applied
source = GetActiveSource()


#### Create a 'Table To Points' object containing the data from the source
tableToPoints1 = TableToPoints(registrationName='Spheres DataTable', Input=source, XColumn='X', YColumn='Y', ZColumn='Z')


#### Create a 'Glyph' object of type 'Sphere' to represent the data points
# Create a sphere glyph with position array X,Y,Z (input from object tableToPoints1)
glyph1 = Glyph(registrationName='Spheres View', Input=tableToPoints1, GlyphType='Sphere')
# Scale sphere diameter with column 'R' of the input data
glyph1.ScaleArray = ['POINTS','R']
# Since column 'R' contain radius values, scale it by a factor 2 to obtain diameter
glyph1.ScaleFactor = 2.0
# Display all data points
glyph1.GlyphMode = 'All Points'
# Increase resolution for a smoother rendering
glyph1.GlyphType.ThetaResolution = 20
glyph1.GlyphType.PhiResolution = 20
# All other properties are left to their default value


#### Display the glyph in a RenderView
# Get active view
renderView1 = GetActiveView() 
# Set object 'glyph1' to active source
SetActiveSource(glyph1)
# Show data in view
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
# Reset view to fit the input data
renderView1.ResetCamera(False)


if 'ClusterIndex' in tableToPoints1.PointData.keys():
    #### Color each sphere by its ClusterIndex properties, using a custom categorical colormap, starting from index 1 (out-of-range indexes appearing in white)
    # Set scalar coloring with a colormap separated from the rest of the view
    ColorBy(glyph1Display, ('POINTS','ClusterIndex'), separate=True)
    # Get the color transfer function for 'ClusterIndex'
    customLUT = GetColorTransferFunction('ClusterIndex', glyph1Display)
    # Interpret cluster values as categories (in contrast to scalar range)
    customLUT.InterpretValuesAsCategories = 1
    # Manually rescale the transfert function to start from 0
    r = tableToPoints1.PointData['ClusterIndex'].GetRange()
    if r[0]<1:
        listAn = []
        for i in range(0,int(r[1])+1):
            listAn.append(str(i))
            listAn.append(str(i))
        customLUT.Annotations = listAn
    # Set color for out-of-range objects to white
    customLUT.UseBelowRangeColor = 0
    customLUT.BelowRangeColor = [1.0, 1.0, 1.0]
    
    
    #### Apply the categorical colormap 'SashaTrubetskoy' or, if not loaded, the default 'KAAMS' preset. 
    ## check if preset 'SashaTrubetskoy' is already loaded
    setup='SashaTrubetskoy'
    if setup not in GetLookupTableNames(): # list of all loaded presets
        setup='KAAMS'
        ## if not, try to load it
        #try:
        #    path = os.path.dirname(__file__)
        #    ImportPresets(filename=path+'/'+setup+'.json')
        ## and eventually fall back on 'KAAMS' 
        #except:
        #    setup='KAAMS'
    # apply preset 
    customLUT.ApplyPreset(setup, True)
    
    
    #### Display color legend with black labels over white background
    # Get color bar
    customLUTColorBar = GetScalarBar(customLUT, renderView1)
    # Set color bar to visible
    customLUTColorBar.Visibility = 1
    customLUTColorBar.WindowLocation = 'Any Location'
    customLUTColorBar.Position = [0.9, 0.25]
    customLUTColorBar.ScalarBarLength = 0.5
    
    customLUTColorBar.Title = ''
    customLUTColorBar.ComponentTitle = ''
    customLUTColorBar.LabelFontFamily = 'Times'
    customLUTColorBar.LabelFontSize = 16
    customLUTColorBar.LabelBold = 1
    customLUTColorBar.LabelColor = [0.0, 0.0, 0.0] # Set labels text color to black
    customLUTColorBar.TextPosition = 'Ticks left/bottom, annotations right/top'
else:
    #### Color all spheres in white
    glyph1Display.AmbientColor = [1.0, 1.0, 1.0]
    glyph1Display.DiffuseColor = [1.0, 1.0, 1.0]

