import autolens as al
import autolens.plot as aplt







image_plane_grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)







#Mass Profiles
mass_profile1 = al.mp.PointMass()
mass_profile2 = al.mp.PointMass2()

#Lens Galaxies
lens_galaxy1 = al.Galaxy(redshift=0.5, mass=mass_profile1)
lens_galaxy2 = al.Galaxy(redshift=0.5, mass=mass_profile2)

#Source Galaxy

light_profile = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)
source_galaxy = al.Galaxy(redshift=1.0, light=light_profile)


#Image Planes
image_plane1 = al.Plane(galaxies=[lens_galaxy1])
image_plane2 = al.Plane(galaxies=[lens_galaxy2])


#Deflections angles
print('PointMass 1')
print(mass_profile1.deflections_from_grid(grid = image_plane_grid))
print('\n')
print('PointMass 2')
print(mass_profile2.deflections_from_grid(grid = image_plane_grid))


#Source Grid traces

source_plane_grid1 = image_plane1.traced_grid_from_grid(grid=image_plane_grid)
source_plane_grid2 = image_plane2.traced_grid_from_grid(grid=image_plane_grid)


#Source Plane

source_plane = al.Plane(galaxies=[source_galaxy])


#Plot Grids of source plane
'''
aplt.Plane.plane_grid(
    plane=source_plane,
    grid=source_plane_grid1,
    axis_limits=[-5, 5, -5, 5],
    plotter=aplt.Plotter(labels=aplt.Labels(title="PointMass 1")),
)

aplt.Plane.plane_grid(
    plane=source_plane,
    grid=source_plane_grid2,
    
    plotter=aplt.Plotter(labels=aplt.Labels(title="PointMass 2")),
)

'''
aplt.Plane.image_and_source_plane_subplot(
    image_plane=image_plane2,
    source_plane=source_plane,
    axis_limits=[-5, 5, -5, 5],
    grid=image_plane_grid,
    indexes=[
        range(0, 500),
        range(3000,3100),
        range(9500,10000),
        range(7500, 7600),
        [1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250],
        [6250, 8550, 8450, 8350, 8250, 8150, 8050, 7950, 7850, 7750],
    ],
)

#Lensing image
aplt.Plane.image(plane=source_plane, grid=source_plane_grid1)
aplt.Plane.image(plane=source_plane, grid=source_plane_grid2)


#Source Plane
aplt.Plane.plane_image(
    plane=source_plane, grid=source_plane_grid1, include=aplt.Include(grid=True)
)

aplt.Plane.plane_image(
    plane=source_plane, grid=source_plane_grid2, include=aplt.Include(grid=True)
)



