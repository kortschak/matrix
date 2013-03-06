// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package matrix provides basic linear algebra operations.
//
// Note that in all interfaces, a c parameter is the recipient of the data and may be
// nil, in which case a new matrix is allocated. When these methods return a Matrix, this
// Matrix will be identical to c when the call returns.
package matrix

// Matrix is the basic matrix interface type.
type Matrix interface {
	// Dims returns the dimensions of a Matrix.
	Dims() (r int, c int)

	// At returns the value of a matrix element at (r, c). It will panic if r or c are
	// out of bounds for the matrix.
	At(r, c int) float64
}

// Mutable is a matrix interface type that allows elements to be altered.
type Mutable interface {
	// Set alters the matrix element at (r, c) to v. It will panic if r or c are out of
	// bounds for the matrix.
	Set(r, c int, v float64)

	Matrix
}

// A Vectorer can return rows and columns of the represented matrix.
type Vectorer interface {
	// Row returns a slice of float64 for the row specified. It will panic if the index
	// is out of bounds. If the call requires a copy and c is long enough to hold the row
	// it will be used and returned.
	Row(int, c []float64) []float64

	// Col returns a slice of float64 for the column specified. It will panic if the index
	// is out of bounds. If the call requires a copy and c is long enough to hold the column
	// it will be used and returned.
	Col(int, c []float64) []float64
}

// A Cloner can make a copy of its elements into the mutable matrix c.
type Cloner interface {
	Clone(c Mutable) Matrix
}

// A Normer returns the specified matrix norm, o of the matrix represented by the receiver.
// ErrNormOrder is returned if o is not valid.
type Normer interface {
	Norm(o int) (float64, error)
}

// A Transposer can transpose the matrix represented by the receiver, placing the elements into c.
// If the concrete value of c is the receiver a new Mutable of the same type is allocated.
type Transposer interface {
	T(c Mutable) Matrix
}

// A Deter can return the determinant of the represented matrix.
type Deter interface {
	Det() float64
}

// An Inver can calculate the inverse of the matrix represented by the receiver. ErrSingular is
// returned if there is not inverse of the matrix.
type Inver interface {
	Inv(c Mutable) (Matrix, error)
}

// An Adder can add the matrices represented by b and the receiver, placing the result in c. Add
// will panic if the two matrices do not have the same shape.
type Adder interface {
	Add(b Matrix, c Mutable) Matrix
}

// A Suber can subtract the matrix represented by b from the receiver, placing the result in c. Sub
// will panic if the two matrices do not have the same shape.
type Suber interface {
	Sub(b Matrix, c Mutable) Matrix
}

// An ElemMuler can perform element-wise multiplication of the matrices represented by b and the
// receive, placing the result in c. MulEmen will panic if the two matrices do not have the same shape.
type ElemMuler interface {
	MulElem(b Matrix, c Mutable) Matrix
}

// An Equaler can compare the matrices represented by b and the receiver. Matrices with non-equal shapes
// are not equal.
type Equaler interface {
	Equals(b Matrix) bool
}

// An ApproxEqualer can compare the matrices represented by b and the receiver, with tolerance for
// element-wise equailty specified by epsilon. Matrices with non-equal shapes are not equal.
type ApproxEqualer interface {
	EqualsApprox(b Matrix, epsilon float64) bool
}

// A Scaler can perform scalar multiplication of the matrix represented by the receiver with f,
// placing the result in c.
type Scaler interface {
	Scalar(f float64, c Mutable) Matrix
}

// A Sumer can return the sum of elements of the matrix represented by the receiver.
type Sumer interface {
	Sum() float64
}

// A Muler can determine the matrix product of b and the receiver, placing the result in c.
// If the number of column of the receiver does not equal the number of rows in b, Mul will panic.
// If the concrete value of c is the receiver a new Mutable of the same type is allocated.
type Muler interface {
	Mul(b, c Mutable) Matrix
}

// A Dotter can determine the inner product of the elements of the receiver and b. If the shapes of
// the two matrices differ, Dot will panic.
type Dotter interface {
	Dot(b Matrix) float64
}

// A Stacker can create the stacked matrix of the receiver with b, where b is placed in the higher
// indexed rows. The result of stacking is placed in c. Stack will panic if the two input matrices do
// not have the same number of columns.
type Stacker interface {
	Stack(b, c Mutable) Matrix
}

// An Augmenter can create the augmented matrix of the receiver with b, where b is placed in the higher
// indexed columns. The result of augmentation is placed in c. Augment will panic if the two input
// matrices do not have the same number of rows.
type Augmenter interface {
	Augment(b, c Mutable) Matrix
}

// An ApplyFunc takes a row/col index and element value and returns some function of that tuple.
type ApplyFunc func(r, c int, v float64) float64

// An Applyer can apply an Applyfunc f to each of its elements, placing the resulting matrix in c.
type Applyer interface {
	Apply(f ApplyFunc, c Mutable) Matrix
}

// A Tracer can return the trace of the matrix represented by the receiver. Trace will panic if the
// matrix is not square.
type Tracer interface {
	Trace() float64
}

// A Uer can return the upper triangular matrix of the receiver, placing the result in c. If the
// concrete value of c is the receiver, the lower residue is zeroed.
type Uer interface {
	U(c Mutable) Matrix
}

// An Ler can return the lower triangular matrix of the receiver, placing the result in c. If the
// concrete value of c is the receiver, the upper residue is zeroed.
type Ler interface {
	L(c Mutable) Matrix
}

// BlasMatrix represents a cblas native representation of a matrix.
type BlasMatrix struct {
	Rows, Cols int
	Stride     int
	Data       []float64
}

// Matrix converts a BlasMatrix to a Matrix.
func (b BlasMatrix) Matrix() Matrix

// A Blasser can return a BlasMatrix representation of the receiver. Changes to the BlasMatrix.Data
// slice will be reflected in the original matrix, changes to the Rows, Cols and Stride fields will not.
type Blasser interface {
	BlasMatrix() BlasMatrix
}

// A Panicker is a function that returns a matrix and may panic.
type Panicker func() Matrix

// Maybe will recover a panic with a type matrix.Error from fn, and return this error.
// Any other error is re-panicked.
func Maybe(fn Panicker) (m Matrix, err error) {
	defer func() {
		if r := recover(); r != nil {
			var ok bool
			if err, ok = r.(Error); ok {
				return
			}
			panic(r)
		}
	}()
	return fn(), nil
}

// A FloatPanicker is a function that returns a float64 and may panic.
type FloatPanicker func() float64

// MaybeFloat will recover a panic with a type matrix.Error from fn, and return this error.
// Any other error is re-panicked.
func MaybeFloat(fn FloatPanicker) (f float64, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(Error); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	return fn(), nil
}

// Must can be used to wrap a function returning a matrix and an error.
// If the returned error is not nil, Must will panic.
func Must(m Matrix, err error) Matrix {
	if err != nil {
		panic(err)
	}
	return m
}

// Type Error represents matrix package errors. These errors can be recovered by Maybe wrappers.
type Error string

func (err Error) Error() string { return string(err) }

const (
	ErrIndexOutOfRange = Error("matrix: index out of range")
	ErrZeroLength      = Error("matrix: zero length in matrix definition")
	ErrRowLength       = Error("matrix: row length mismatch")
	ErrColLength       = Error("matrix: col length mismatch")
	ErrSquare          = Error("matrix: expect square matrix")
	ErrNormOrder       = Error("matrix: invalid norm order for matrix")
	ErrSingular        = Error("matrix: matrix is singular")
	ErrShape           = Error("matrix: dimension mismatch")
	ErrIllegalStride   = Error("matrix: illegal stride")
	ErrPivot           = Error("matrix: malformed pivot list")
)
