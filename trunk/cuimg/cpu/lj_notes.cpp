

Structure lj<T, S, N>:
     vecteur de T
     acces à une echelle donnée
     acces à une dérivée.
     => operator() (unsigned s, unsigned i, unsigned j)

taille statique ou dynamique?? -> statique.


// Construction:
// application du banc de filtres:
//          lj_fill(in, out);
//          lj_convolve_rows(out, tmp);
//          lj_convolve_cols(tmp, out);
//     -> Retourne une image de lj<T>

//Exctraction d'une composante:
lj_extract(ljs, out);

struct localjets
