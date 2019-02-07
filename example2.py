from cspd import variable_stoichiometry_generator

variable_stoichiometry_generator(
    symbols='TiO',
    stoichiometry=[
        [1,2],[1,3],[1,4]
    ],
    # lw=True,
    # format='cif',
    # fu=[1, 4],
    cspd_file='CSPD.db',
    # sgn=[71,71],
    # to_primitive=True,
    # mindis=[
    #     [1.85,  1.113],
    #     [1.113, 1.202]],
    # nstr=20,
    # clean=True,
    # maxatomn=56,
)