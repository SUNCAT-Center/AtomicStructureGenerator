from cspd import atomic_structure_generator


structure_list = atomic_structure_generator(
    symbols='TiO2',
    # lw=True,
    # format='cif',
    # fu=[1, 4],
    cspd_file='CSPD.db',
    # sgn=[1,225],
    # ndensity=0.071,
    # volume=49.0,
    # mindis=[
    #     [1.85,  1.113],
    #     [1.113, 1.202]],
    # nstr=300,
    # maxatomn=60,
)

exit()