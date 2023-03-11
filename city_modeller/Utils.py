
def distancia_mas_cercano(geom,parques = parques_multi):
    par = nearest_points(geom,parques)
    return par[0].distance(par[1])

def pob_a_distancia(minutos,radios=radios_p):
    #velocidad de caminata 5km/h
    metros = minutos*5/60*1000
    radios['metros'] = radios.distancia <= metros
    tabla = radios.loc[:,['metros','TOTAL_POB']].groupby('metros').sum()
    return round(tabla['TOTAL_POB'][True] / tabla['TOTAL_POB'].sum()* 100)

def pob_a_distancia_area(area, minutos = 5,radios=radios_modificable):
    parques_multi = MultiPoint([i for i in parques_p.loc[parques_p.loc[:,'area'] > area,'geometry']])
    def distancia_mas_cercano(geom,parques = parques_multi):
    par = nearest_points(geom,parques)
    return par[0].distance(par[1])
    radios['distancia'] = radios.geometry.map(distancia_mas_cercano)
    #velocidad de caminata 5km/h
    metros = minutos*(5/(60*1000))
    radios['metros'] = radios.distancia <= metros
    tabla = radios.loc[:,['metros','TOTAL_VIV']].groupby('metros').sum()
    return round(tabla['TOTAL_VIV'][True] / tabla['TOTAL_VIV'].sum()* 100)


def plot_curva_pob_min_cam():
    minutos = range(1,21)
    prop = [pob_a_distancia(minuto) for minuto in minutos]
    f, ax = plt.subplots(1,figsize=(8,6))
    ax.plot(minutos,prop,'darkgreen')
    ax.set_title('Porcentaje de población en CABA según minutos de caminata a un parque público')
    ax.set_xlabel('Minutos de caminata a un parque público')
    ax.set_ylabel('Porcentaje de población de la CABA');
    #f.savefig('porcentajeXminutos.png')