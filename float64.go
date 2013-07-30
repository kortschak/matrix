package matrix

import (
	"github.com/gonum/blas"
	"math"
)

var blasEngine blas.Float64

func Register(b blas.Float64) { blasEngine = b }

var blasOrder = blas.RowMajor

func Order(o blas.Order) blas.Order {
	if o == blas.RowMajor || o == blas.ColMajor {
		o, blasOrder = blasOrder, o
		return o
	}
	return blasOrder
}

var (
	matrix *Float64

	_ Matrix       = matrix
	_ Mutable      = matrix
	_ Vectorer     = matrix
	_ VectorSetter = matrix

	_ Cloner      = matrix
	_ Viewer      = matrix
	_ Submatrixer = matrix

	_ Adder     = matrix
	_ Suber     = matrix
	_ Muler     = matrix
	_ Dotter    = matrix
	_ ElemMuler = matrix

	_ Scaler  = matrix
	_ Applyer = matrix

	_ Transposer = matrix

	// _ Deter  = matrix
	// _ Inver  = matrix
	_ Tracer = matrix
	// _ Normer = matrix
	_ Sumer = matrix

	// _ Uer = matrix
	// _ Ler = matrix

	// _ Stacker   = matrix
	// _ Augmenter = matrix

	_ Equaler       = matrix
	_ ApproxEqualer = matrix

	_ BlasLoader = matrix
	_ Blasser    = matrix
)

type Float64 struct {
	mat BlasMatrix
}

func (m *Float64) LoadBlas(b BlasMatrix) { m.mat = b }

func (m *Float64) BlasMatrix() BlasMatrix { return m.mat }

func (m *Float64) isZero() bool {
	return m.mat.Cols == 0 || m.mat.Rows == 0
}

func realloc(f []float64, l int) []float64 {
	if l < cap(f) {
		return f[:l]
	}
	return make([]float64, l)
}

func (m *Float64) At(r, c int) float64 {
	switch m.mat.Order {
	case blas.RowMajor:
		return m.mat.Data[r*m.mat.Stride+c]
	case blas.ColMajor:
		return m.mat.Data[c*m.mat.Stride+r]
	default:
		panic(ErrIllegalOrder)
	}
}

func (m *Float64) Set(r, c int, v float64) {
	switch m.mat.Order {
	case blas.RowMajor:
		m.mat.Data[r*m.mat.Stride+c] = v
	case blas.ColMajor:
		m.mat.Data[c*m.mat.Stride+r] = v
	default:
		panic(ErrIllegalOrder)
	}
}

func (m *Float64) Dims() (r, c int) { return m.mat.Rows, m.mat.Cols }

func (m *Float64) Col(c int, col []float64) []float64 {
	if c >= m.mat.Cols || c < 0 {
		panic(ErrIndexOutOfRange)
	}

	if len(col) < m.mat.Rows {
		col = make([]float64, m.mat.Rows)
	}
	switch m.mat.Order {
	case blas.RowMajor:
		blasEngine.Dcopy(m.mat.Rows, m.mat.Data[c:], m.mat.Stride, col, 1)
	case blas.ColMajor:
		copy(col, m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows])
	default:
		panic(ErrIllegalOrder)
	}

	return col
}

func (m *Float64) SetCol(c int, v []float64) {
	if c >= m.mat.Cols || c < 0 {
		panic(ErrIndexOutOfRange)
	}

	if len(v) != m.mat.Rows {
		panic(ErrShape)
	}
	switch m.mat.Order {
	case blas.RowMajor:
		blasEngine.Dcopy(m.mat.Rows, v, 1, m.mat.Data[c:], m.mat.Stride)
	case blas.ColMajor:
		copy(m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows], v)
	default:
		panic(ErrIllegalOrder)
	}
}

func (m *Float64) Row(r int, row []float64) []float64 {
	if r >= m.mat.Rows || r < 0 {
		panic(ErrIndexOutOfRange)
	}

	if len(row) < m.mat.Cols {
		row = make([]float64, m.mat.Cols)
	}
	switch m.mat.Order {
	case blas.RowMajor:
		copy(row, m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols])
	case blas.ColMajor:
		blasEngine.Dcopy(m.mat.Cols, m.mat.Data[r:], m.mat.Stride, row, 1)
	default:
		panic(ErrIllegalOrder)
	}

	return row
}

func (m *Float64) SetRow(r int, v []float64) {
	if r >= m.mat.Rows || r < 0 {
		panic(ErrIndexOutOfRange)
	}

	if len(v) != m.mat.Cols {
		panic(ErrShape)
	}
	switch m.mat.Order {
	case blas.RowMajor:
		copy(m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols], v)
	case blas.ColMajor:
		blasEngine.Dcopy(m.mat.Cols, v, 1, m.mat.Data[r:], m.mat.Stride)
	default:
		panic(ErrIllegalOrder)
	}
}

// View returns a view on the receiver.
func (m *Float64) View(i, j, r, c int) Matrix {
	v := *m
	switch m.mat.Order {
	case blas.RowMajor:
		v.mat.Data = m.mat.Data[i*m.mat.Stride+j : (i+r-1)*m.mat.Stride+(j+c)]
	case blas.ColMajor:
		v.mat.Data = m.mat.Data[i+j*m.mat.Stride : (i+r)+(j+c-1)*m.mat.Stride]
	default:
		panic(ErrIllegalOrder)
	}
	return &v
}

func (m *Float64) Submatrix(a Matrix, i, j, r, c int) {
	// This is probably a bad idea, but for the moment, we do it.
	m.Clone(m.View(i, j, r, c))
}

func (m *Float64) Clone(a Matrix) {
	r, c := a.Dims()
	m.mat = BlasMatrix{
		Order: blasOrder,
		Rows:  r,
		Cols:  c,
	}
	data := make([]float64, r*c)
	switch a := a.(type) {
	case *Float64:
		switch blasOrder {
		case blas.RowMajor:
			for i := 0; i < r; i++ {
				copy(data[i*c:(i+1)*c], a.mat.Data[i*a.mat.Stride:i*a.mat.Stride+r])
			}
			m.mat.Stride = c
		case blas.ColMajor:
			for i := 0; i < c; i++ {
				copy(data[i*r:(i+1)*r], a.mat.Data[i*a.mat.Stride:i*a.mat.Stride+c])
			}
			m.mat.Stride = r
		default:
			panic(ErrIllegalOrder)
		}
		m.mat.Data = data
	case Vectorer:
		switch blasOrder {
		case blas.RowMajor:
			for i := 0; i < r; i++ {
				a.Row(i, data[i*c:(i+1)*c])
			}
			m.mat.Stride = c
		case blas.ColMajor:
			for i := 0; i < c; i++ {
				a.Col(i, data[i*r:(i+1)*r])
			}
			m.mat.Stride = r
		default:
			panic(ErrIllegalOrder)
		}
		m.mat.Data = data
	default:
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				m.Set(r, c, a.At(r, c))
			}
		}
	}
}

func (m *Float64) Min() float64 {
	var i, j int
	switch m.mat.Order {
	case blas.RowMajor:
		i, j = m.mat.Rows, m.mat.Cols
	case blas.ColMajor:
		i, j = m.mat.Cols, m.mat.Rows
	default:
		panic(ErrIllegalOrder)
	}
	min := m.mat.Data[0]
	for k := 0; k < i; k++ {
		for _, v := range m.mat.Data[k*m.mat.Stride : k*m.mat.Stride+j] {
			min = math.Min(min, v)
		}
	}
	return min
}

func (m *Float64) Max() float64 {
	var i, j int
	switch m.mat.Order {
	case blas.RowMajor:
		i, j = m.mat.Rows, m.mat.Cols
	case blas.ColMajor:
		i, j = m.mat.Cols, m.mat.Rows
	default:
		panic(ErrIllegalOrder)
	}
	max := m.mat.Data[0]
	for k := 0; k < i; k++ {
		for _, v := range m.mat.Data[k*m.mat.Stride : k*m.mat.Stride+j] {
			max = math.Max(max, v)
		}
	}
	return max
}

func (m *Float64) Trace() float64 {
	if m.mat.Rows != m.mat.Cols {
		panic(ErrSquare)
	}
	var t float64
	for i := 0; i < len(m.mat.Data); i += m.mat.Stride + 1 {
		t += m.mat.Data[i]
	}
	return t
}

func (m *Float64) Add(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	var k, l int
	if m.isZero() {
		m.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ar,
			Cols:  ac,
			Data:  realloc(m.mat.Data, ar*ac),
		}
		switch blasOrder {
		case blas.RowMajor:
			m.mat.Stride, k, l = ac, ar, ac
		case blas.ColMajor:
			m.mat.Stride, k, l = ar, ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	} else {
		switch blasOrder {
		case blas.RowMajor:
			k, l = ar, ac
		case blas.ColMajor:
			k, l = ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	}

	// This is the fast path; both are really BlasMatrix types.
	if a, ok := a.(*Float64); ok {
		if b, ok := b.(*Float64); ok {
			if a.mat.Order != blasOrder || b.mat.Order != blasOrder {
				panic(ErrIllegalOrder)
			}
			for ja, jb, jm := 0, 0, 0; ja < k*a.mat.Stride; ja, jb, jm = ja+a.mat.Stride, jb+b.mat.Stride, jm+m.mat.Stride {
				for i, v := range a.mat.Data[ja : ja+l] {
					m.mat.Data[i+jm] = v + b.mat.Data[i+jb]
				}
			}
			return
		}
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			switch blasOrder {
			case blas.RowMajor:
				rowa := make([]float64, ac)
				rowb := make([]float64, bc)
				for r := 0; r < ar; r++ {
					a.Row(r, rowa)
					b.Row(r, rowb)
					for i, v := range rowb {
						rowa[i] += v
					}
					copy(m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols], rowa)
				}
			case blas.ColMajor:
				cola := make([]float64, ar)
				colb := make([]float64, br)
				for c := 0; c < ac; c++ {
					a.Col(c, cola)
					b.Col(c, colb)
					for i, v := range colb {
						cola[i] += v
					}
					copy(m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows], cola)
				}
			default:
				panic(ErrIllegalOrder)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.Set(r, c, a.At(r, c)+b.At(r, c))
		}
	}
}

func (m *Float64) Sub(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	var k, l int
	if m.isZero() {
		m.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ar,
			Cols:  ac,
			Data:  realloc(m.mat.Data, ar*ac),
		}
		switch blasOrder {
		case blas.RowMajor:
			m.mat.Stride, k, l = ac, ar, ac
		case blas.ColMajor:
			m.mat.Stride, k, l = ar, ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	} else {
		switch blasOrder {
		case blas.RowMajor:
			k, l = ar, ac
		case blas.ColMajor:
			k, l = ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	}

	// This is the fast path; both are really BlasMatrix types.
	if a, ok := a.(*Float64); ok {
		if b, ok := b.(*Float64); ok {
			if a.mat.Order != blasOrder || b.mat.Order != blasOrder {
				panic(ErrIllegalOrder)
			}
			for ja, jb, jm := 0, 0, 0; ja < k*a.mat.Stride; ja, jb, jm = ja+a.mat.Stride, jb+b.mat.Stride, jm+m.mat.Stride {
				for i, v := range a.mat.Data[ja : ja+l] {
					m.mat.Data[i+jm] = v - b.mat.Data[i+jb]
				}
			}
			return
		}
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			switch blasOrder {
			case blas.RowMajor:
				rowa := make([]float64, ac)
				rowb := make([]float64, bc)
				for r := 0; r < ar; r++ {
					a.Row(r, rowa)
					b.Row(r, rowb)
					for i, v := range rowb {
						rowa[i] -= v
					}
					copy(m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols], rowa)
				}
			case blas.ColMajor:
				cola := make([]float64, ar)
				colb := make([]float64, br)
				for c := 0; c < ac; c++ {
					a.Col(c, cola)
					b.Col(c, colb)
					for i, v := range colb {
						cola[i] -= v
					}
					copy(m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows], cola)
				}
			default:
				panic(ErrIllegalOrder)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.Set(r, c, a.At(r, c)-b.At(r, c))
		}
	}
}

func (m *Float64) MulElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(ErrShape)
	}

	var k, l int
	if m.isZero() {
		m.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ar,
			Cols:  ac,
			Data:  realloc(m.mat.Data, ar*ac),
		}
		switch blasOrder {
		case blas.RowMajor:
			m.mat.Stride, k, l = ac, ar, ac
		case blas.ColMajor:
			m.mat.Stride, k, l = ar, ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	} else {
		switch blasOrder {
		case blas.RowMajor:
			k, l = ar, ac
		case blas.ColMajor:
			k, l = ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	}

	// This is the fast path; both are really BlasMatrix types.
	if a, ok := a.(*Float64); ok {
		if b, ok := b.(*Float64); ok {
			if a.mat.Order != blasOrder || b.mat.Order != blasOrder {
				panic(ErrIllegalOrder)
			}
			for ja, jb, jm := 0, 0, 0; ja < k*a.mat.Stride; ja, jb, jm = ja+a.mat.Stride, jb+b.mat.Stride, jm+m.mat.Stride {
				for i, v := range a.mat.Data[ja : ja+l] {
					m.mat.Data[i+jm] = v * b.mat.Data[i+jb]
				}
			}
			return
		}
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			switch blasOrder {
			case blas.RowMajor:
				rowa := make([]float64, ac)
				rowb := make([]float64, bc)
				for r := 0; r < ar; r++ {
					a.Row(r, rowa)
					b.Row(r, rowb)
					for i, v := range rowb {
						rowa[i] *= v
					}
					copy(m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols], rowa)
				}
			case blas.ColMajor:
				cola := make([]float64, ar)
				colb := make([]float64, br)
				for c := 0; c < ac; c++ {
					a.Col(c, cola)
					b.Col(c, colb)
					for i, v := range colb {
						cola[i] *= v
					}
					copy(m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows], cola)
				}
			default:
				panic(ErrIllegalOrder)
			}
			return
		}
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.Set(r, c, a.At(r, c)*b.At(r, c))
		}
	}
}

func (m *Float64) Dot(b Matrix) float64 {
	mr, mc := m.Dims()
	br, bc := b.Dims()

	if mr != br || mc != bc {
		panic(ErrShape)
	}

	var k, l int
	switch blasOrder {
	case blas.RowMajor:
		k, l = mr, mc
	case blas.ColMajor:
		k, l = mc, mr
	default:
		panic(ErrIllegalOrder)
	}

	var d float64

	// This is the fast path; both are really BlasMatrix types.
	if b, ok := b.(*Float64); ok {
		if m.mat.Order != blasOrder || b.mat.Order != blasOrder {
			panic(ErrIllegalOrder)
		}
		for jm, jb := 0, 0; jm < k*m.mat.Stride; jm, jb = jm+m.mat.Stride, jb+b.mat.Stride {
			for i, v := range m.mat.Data[jm : jm+l] {
				d += v * b.mat.Data[i+jb]
			}
		}
		return d
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if b, ok := b.(Vectorer); ok {
		switch blasOrder {
		case blas.RowMajor:
			row := make([]float64, bc)
			for r := 0; r < br; r++ {
				for i, v := range b.Row(r, row) {
					d += m.mat.Data[r*m.mat.Stride+i] * v
				}
			}
		case blas.ColMajor:
			col := make([]float64, br)
			for c := 0; c < bc; c++ {
				for i, v := range b.Col(c, col) {
					d += m.mat.Data[c*m.mat.Stride+i] * v
				}
			}
		default:
			panic(ErrIllegalOrder)
		}
		return d
	}

	for r := 0; r < mr; r++ {
		for c := 0; c < mc; c++ {
			d += m.At(r, c) * b.At(r, c)
		}
	}
	return d
}

func (m *Float64) Mul(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	var w Float64
	if m != a && m != b {
		w = *m
	}
	if w.isZero() {
		w.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ar,
			Cols:  bc,
			Data:  realloc(w.mat.Data, ar*bc),
		}
		switch blasOrder {
		case blas.RowMajor:
			w.mat.Stride = bc
		case blas.ColMajor:
			w.mat.Stride = br
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != w.mat.Rows || bc != w.mat.Cols {
		panic(ErrShape)
	}

	// This is the fast path; both are really BlasMatrix types.
	if a, ok := a.(*Float64); ok {
		if b, ok := b.(*Float64); ok {
			if a.mat.Order != blasOrder || b.mat.Order != blasOrder {
				panic(ErrIllegalOrder)
			}
			blasEngine.Dgemm(
				blasOrder,
				blas.NoTrans, blas.NoTrans,
				ar, bc, ac,
				1.,
				a.mat.Data, a.mat.Stride,
				b.mat.Data, b.mat.Stride,
				0.,
				w.mat.Data, w.mat.Stride)
			*m = w
			return
		}
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if a, ok := a.(Vectorer); ok {
		if b, ok := b.(Vectorer); ok {
			row := make([]float64, ac)
			col := make([]float64, br)
			for r := 0; r < ar; r++ {
				for c := 0; c < bc; c++ {
					switch blasOrder {
					case blas.RowMajor:
						w.mat.Data[r*w.mat.Stride+w.mat.Cols] = blasEngine.Ddot(ac, a.Row(r, row), 1, b.Col(c, col), 1)
					case blas.ColMajor:
						w.mat.Data[c*w.mat.Stride+w.mat.Rows] = blasEngine.Ddot(ac, a.Row(r, row), 1, b.Col(c, col), 1)
					default:
						panic(ErrIllegalOrder)
					}
				}
			}
			*m = w
			return
		}
	}

	row := make([]float64, ac)
	for r := 0; r < ar; r++ {
		for i := range row {
			row[i] = a.At(r, i)
		}
		for c := 0; c < bc; c++ {
			var v float64
			row := make([]float64, ac)
			for i, e := range row {
				v += e * b.At(i, c)
			}
			switch blasOrder {
			case blas.RowMajor:
				w.mat.Data[r*w.mat.Stride+w.mat.Cols] = v
			case blas.ColMajor:
				w.mat.Data[c*w.mat.Stride+w.mat.Rows] = v
			default:
				panic(ErrIllegalOrder)
			}
		}
	}
	*m = w
}

func (m *Float64) Scale(f float64, a Matrix) {
	ar, ac := a.Dims()

	var k, l int
	if m.isZero() {
		m.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ar,
			Cols:  ac,
			Data:  realloc(m.mat.Data, ar*ac),
		}
		switch blasOrder {
		case blas.RowMajor:
			m.mat.Stride, k, l = ac, ar, ac
		case blas.ColMajor:
			m.mat.Stride, k, l = ar, ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	} else {
		switch blasOrder {
		case blas.RowMajor:
			k, l = ar, ac
		case blas.ColMajor:
			k, l = ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	}

	if a, ok := a.(*Float64); ok {
		if a.mat.Order != blasOrder {
			panic(ErrIllegalOrder)
		}
		for ja, jm := 0, 0; ja < k*a.mat.Stride; ja, jm = ja+a.mat.Stride, jm+m.mat.Stride {
			for i, v := range a.mat.Data[ja : ja+l] {
				m.mat.Data[i+jm] = v * f
			}
		}
		return
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if a, ok := a.(Vectorer); ok {
		switch blasOrder {
		case blas.RowMajor:
			row := make([]float64, ac)
			for r := 0; r < ar; r++ {
				a.Row(r, row)
				for i, v := range row {
					row[i] = f * v
				}
				copy(m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols], row)
			}
		case blas.ColMajor:
			col := make([]float64, ar)
			for c := 0; c < ac; c++ {
				a.Col(c, col)
				for i, v := range col {
					col[i] = f * v
				}
				copy(m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows], col)
			}
		default:
			panic(ErrIllegalOrder)
		}
		return
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.Set(r, c, f*a.At(r, c))
		}
	}
}

func (m *Float64) Apply(f ApplyFunc, a Matrix) {
	ar, ac := a.Dims()

	var k, l int
	if m.isZero() {
		m.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ar,
			Cols:  ac,
			Data:  realloc(m.mat.Data, ar*ac),
		}
		switch blasOrder {
		case blas.RowMajor:
			m.mat.Stride, k, l = ac, ar, ac
		case blas.ColMajor:
			m.mat.Stride, k, l = ar, ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != m.mat.Rows || ac != m.mat.Cols {
		panic(ErrShape)
	} else {
		switch blasOrder {
		case blas.RowMajor:
			k, l = ar, ac
		case blas.ColMajor:
			k, l = ac, ar
		default:
			panic(ErrIllegalOrder)
		}
	}

	if a, ok := a.(*Float64); ok {
		if a.mat.Order != blasOrder {
			panic(ErrIllegalOrder)
		}
		var r, c int
		for j, ja, jm := 0, 0, 0; ja < k*a.mat.Stride; j, ja, jm = j+1, ja+a.mat.Stride, jm+m.mat.Stride {
			for i, v := range a.mat.Data[ja : ja+l] {
				if blasOrder == blas.RowMajor {
					r, c = i, j
				} else {
					r, c = j, i
				}
				m.mat.Data[i+jm] = f(r, c, v)
			}
		}
		return
	}

	// Progressively worse runtime cases... not all of them - there are four, margin width yadda yadda.
	if a, ok := a.(Vectorer); ok {
		switch blasOrder {
		case blas.RowMajor:
			row := make([]float64, ac)
			for r := 0; r < ar; r++ {
				a.Row(r, row)
				for i, v := range row {
					row[i] = f(r, i, v)
				}
				copy(m.mat.Data[r*m.mat.Stride:r*m.mat.Stride+m.mat.Cols], row)
			}
		case blas.ColMajor:
			col := make([]float64, ar)
			for c := 0; c < ac; c++ {
				a.Col(c, col)
				for i, v := range col {
					col[i] = f(i, c, v)
				}
				copy(m.mat.Data[c*m.mat.Stride:c*m.mat.Stride+m.mat.Rows], col)
			}
		default:
			panic(ErrIllegalOrder)
		}
		return
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.Set(r, c, f(r, c, a.At(r, c)))
		}
	}
}

func (m *Float64) T(a Matrix) {
	ar, ac := a.Dims()

	var w Float64
	if m != a {
		w = *m
	}
	if w.isZero() {
		w.mat = BlasMatrix{
			Order: blasOrder,
			Rows:  ac,
			Cols:  ar,
			Data:  realloc(w.mat.Data, ar*ac),
		}
		switch blasOrder {
		case blas.RowMajor:
			w.mat.Stride = ar
		case blas.ColMajor:
			w.mat.Stride = ac
		default:
			panic(ErrIllegalOrder)
		}
	} else if ar != m.mat.Cols || ac != m.mat.Rows {
		panic(ErrShape)
	} else if blasOrder != blas.RowMajor && blasOrder != blas.ColMajor {
		panic(ErrIllegalOrder)
	}
	switch a := a.(type) {
	case *Float64:
		for i := 0; i < ac; i++ {
			for j := 0; j < ar; j++ {
				w.Set(i, j, a.At(j, i))
			}
		}
	default:
		for i := 0; i < ac; i++ {
			for j := 0; j < ar; j++ {
				w.Set(i, j, a.At(j, i))
			}
		}
	}
	*m = w
}

func (m *Float64) Sum() float64 {
	var l int
	switch blasOrder {
	case blas.RowMajor:
		l = m.mat.Cols
	case blas.ColMajor:
		l = m.mat.Rows
	default:
		panic(ErrIllegalOrder)
	}
	var s float64
	for i := 0; i < len(m.mat.Data); i += m.mat.Stride {
		for _, v := range m.mat.Data[i : i+l] {
			s += v
		}
	}
	return s
}

func (m *Float64) Equals(b Matrix) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}

	if b, ok := b.(*Float64); ok {
		var k, l int
		switch blasOrder {
		case blas.RowMajor:
			k, l = br, bc
		case blas.ColMajor:
			k, l = bc, br
		default:
			panic(ErrIllegalOrder)
		}
		for jb, jm := 0, 0; jm < k*m.mat.Stride; jb, jm = jb+b.mat.Stride, jm+m.mat.Stride {
			for i, v := range m.mat.Data[jm : jm+l] {
				if v != b.mat.Data[i+jb] {
					return false
				}
			}
		}
		return true
	}

	if b, ok := b.(Vectorer); ok {
		switch blasOrder {
		case blas.RowMajor:
			rowb := make([]float64, bc)
			for r := 0; r < br; r++ {
				rowm := m.mat.Data[r*m.mat.Stride : r*m.mat.Stride+m.mat.Cols]
				b.Row(r, rowb)
				for i, v := range rowb {
					if rowm[i] != v {
						return false
					}
				}
			}
		case blas.ColMajor:
			colb := make([]float64, br)
			for c := 0; c < bc; c++ {
				colm := m.mat.Data[c*m.mat.Stride : c*m.mat.Stride+m.mat.Rows]
				b.Col(c, colb)
				for i, v := range colb {
					if colm[i] != v {
						return false
					}
				}
			}
		default:
			panic(ErrIllegalOrder)
		}
		return true
	}

	for r := 0; r < br; r++ {
		for c := 0; c < bc; c++ {
			if m.At(r, c) != b.At(r, c) {
				return false
			}
		}
	}
	return true
}

func (m *Float64) EqualsApprox(b Matrix, epsilon float64) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}

	if b, ok := b.(*Float64); ok {
		var k, l int
		switch blasOrder {
		case blas.RowMajor:
			k, l = br, bc
		case blas.ColMajor:
			k, l = bc, br
		default:
			panic(ErrIllegalOrder)
		}
		for jb, jm := 0, 0; jm < k*m.mat.Stride; jb, jm = jb+b.mat.Stride, jm+m.mat.Stride {
			for i, v := range m.mat.Data[jm : jm+l] {
				if math.Abs(v-b.mat.Data[i+jb]) > epsilon {
					return false
				}
			}
		}
		return true
	}

	if b, ok := b.(Vectorer); ok {
		switch blasOrder {
		case blas.RowMajor:
			rowb := make([]float64, bc)
			for r := 0; r < br; r++ {
				rowm := m.mat.Data[r*m.mat.Stride : r*m.mat.Stride+m.mat.Cols]
				b.Row(r, rowb)
				for i, v := range rowb {
					if math.Abs(rowm[i]-v) > epsilon {
						return false
					}
				}
			}
		case blas.ColMajor:
			colb := make([]float64, br)
			for c := 0; c < bc; c++ {
				colm := m.mat.Data[c*m.mat.Stride : c*m.mat.Stride+m.mat.Rows]
				b.Col(c, colb)
				for i, v := range colb {
					if math.Abs(colm[i]-v) > epsilon {
						return false
					}
				}
			}
		default:
			panic(ErrIllegalOrder)
		}
		return true
	}

	for r := 0; r < br; r++ {
		for c := 0; c < bc; c++ {
			if math.Abs(m.At(r, c)-b.At(r, c)) > epsilon {
				return false
			}
		}
	}
	return true
}
