/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl, Kobe Bergmans,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "qpswift_interface.hpp"
#include "casadi/core/casadi_misc.hpp"

using namespace std;
namespace casadi {

  extern "C"
  int CASADI_CONIC_QPSWIFT_EXPORT
  casadi_register_conic_qpswift(Conic::Plugin* plugin) {
    plugin->creator = QpswiftInterface::creator;
    plugin->name = "qpswift";
    plugin->doc = QpswiftInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &QpswiftInterface::options_;
    plugin->deserialize = &QpswiftInterface::deserialize;
    return 0;
  }

  extern "C"
  void CASADI_CONIC_QPSWIFT_EXPORT casadi_load_conic_qpswift() {
    Conic::registerPlugin(casadi_register_conic_qpswift);
  }

  QpswiftInterface::QpswiftInterface(const std::string& name,
                                   const std::map<std::string, Sparsity>& st)
    : Conic(name, st) {

    has_refcount_ = true;
  }

  QpswiftInterface::~QpswiftInterface() {
    clear_mem();
  }

  const Options QpswiftInterface::options_
  = {{&Conic::options_},
     {{"maxit",
       {OT_INT,
        "Maximum iterations."}},
      {"reltol",
       {OT_DOUBLE,
        "Relative tolerance."}},
      {"abstol",
       {OT_DOUBLE,
        "Absolute tolerance."}},
      {"sigma",
       {OT_DOUBLE,
        "Desired sigma value."}},
      {"verbose",
       {OT_INT,
        "Printing level."}}
     }
  };

  void QpswiftInterface::init(const Dict& opts) {
    // Initialize the base classes
    Conic::init(opts);
    
    maxit_ = 0;
    reltol_ = 0.;
    abstol_ = 0.;
    sigma_ = 0.;
    verbose_ = 0;

    // Read options
    for (auto&& op : opts) {
      if (op.first=="maxit") {
        int temp = op.second;
        maxit_ = static_cast<qp_int>(temp);
      } else if (op.first=="reltol") {
        reltol_ = op.second;
      } else if (op.first=="abstol") {
        abstol_ = op.second;
      } else if (op.first=="sigma") {
        sigma_ = op.second;
      } else if (op.first=="verbose") {
        int temp = op.second;
        verbose_ = static_cast<qp_int>(temp);
      }
    }

    // Allocate work vectors
    // TODO(@KobeBergmans): We need to allocate lots of space because the equality constraints size is not known...
    alloc_w(nx_, true); // lbx
    alloc_w(nx_, true); // ubx
    alloc_w(na_, true); // lba
    alloc_w(na_, true); // uba
    alloc_w(H_.nnz(), true); // Ppr
    alloc_w(A_.nnz(), true); // Apr
    alloc_w(A_.nnz(), true); // Gpr
    alloc_w(nx_, true); // c
    alloc_w(2*(nx_+na_), true); // h
    alloc_w(nx_+na_, true); // b
    
    alloc_w(2*(nx_+A_.nnz()), true); // new_data
    alloc_w(A_.nnz(), true); // a_prob_trans
    alloc_w(A_.nnz(), true); // new_data_gA_upper
    alloc_w(A_.nnz(), true); // new_data_gA_lower

    alloc_w(na_, true); // equality_constr_A
    alloc_w(na_, true); // unbounded_upper_constr_A
    alloc_w(na_, true); // unbounded_lower_constr_A
    alloc_w(nx_, true); // equality_constr_x
    alloc_w(nx_, true); // unbounded_upper_constr_x
    alloc_w(nx_, true); // unbounded_lower_constr_x

    alloc_w(nx_+1+na_+1, true); // new_colind_1 (extra large for codegen)
    alloc_w(2*(nx_+1)+na_+1, true); // new_colind_2 (extra large for codegen)
    alloc_w(na_+1, true); // new_colind_3
    alloc_w(nx_+A_.nnz(), true); // new_row_1 (extra large for codegen)
    alloc_w(2*nx_+A_.nnz(), true); // new_row_2 (extra large for codegen)
    alloc_w(A_.nnz(), true); // new_row_3

    // TODO(@KobeBergmans): This is probably too much memory because qp_ints are only longs so they take 4 bytes
    alloc_w(H_.size2()+1, true); // Pjc
    alloc_w(H_.nnz(), true); // Pir
    alloc_w(2*(nx_ + A_.size2()+1), true); // Gjc
    alloc_w(2*(nx_ + A_.nnz()), true); // Gir
    alloc_w(nx_ + A_.size2()+1, true); // Ajc
    alloc_w(nx_ + A_.nnz(), true); // Air

    alloc_iw(2*(na_+nx_)); // casadi_trans
  }

  int QpswiftInterface::init_mem(void* mem) const {
    if (Conic::init_mem(mem)) return 1;

    return 0;
  }

  inline const char* return_status_string(casadi_int status) {
    return "Unknown";
  }

  int QpswiftInterface::
  solve(const double** arg, double** res, casadi_int* iw, double* w, void* mem) const {
    auto m = static_cast<QpswiftMemory*>(mem);

    // Set memory locations and copy some vectors
    double* lbx=w; w += nx_;
    casadi_copy(arg[CONIC_LBX], nx_, lbx);
    double* ubx=w; w += nx_;
    casadi_copy(arg[CONIC_UBX], nx_, ubx);
    double* lba=w; w += na_;
    casadi_copy(arg[CONIC_LBA], na_, lba);
    double* uba=w; w += na_;
    casadi_copy(arg[CONIC_UBA], na_, uba);
    double* Ppr=w; w += H_.nnz();
    casadi_copy(arg[CONIC_H], H_.nnz(), Ppr);
    double* Apr=w; w += A_.nnz();
    double* Gpr=w; w += A_.nnz();
    double* c=w; w += nx_;
    double* h=w; w += 2*(nx_+na_);
    double* b=w; w += nx_+na_;
    double* new_data = w; w += 2*(nx_+A_.nnz());
    double* a_prob_trans = w; w += A_.nnz();
    double* new_data_gA_upper = w; w += A_.nnz();
    double* new_data_gA_lower = w; w += A_.nnz();
    
    casadi_int* w_ci = reinterpret_cast<casadi_int*>(w);
    casadi_int* equality_constr_A = w_ci; w_ci += na_;
    casadi_int* unbounded_lower_constr_A = w_ci; w_ci += na_;
    casadi_int* unbounded_upper_constr_A = w_ci; w_ci += na_;
    casadi_int* equality_constr_x = w_ci; w_ci += nx_;
    casadi_int* unbounded_lower_constr_x = w_ci; w_ci += nx_;
    casadi_int* unbounded_upper_constr_x = w_ci; w_ci += nx_;

    casadi_int* new_colind_1 = w_ci; w_ci += na_+1;
    casadi_int* new_colind_2 = w_ci; w_ci += na_+1;
    casadi_int* new_colind_3 = w_ci; w_ci += na_+1;
    casadi_int* new_row_1 = w_ci; w_ci += A_.nnz();
    casadi_int* new_row_2 = w_ci; w_ci += A_.nnz();
    casadi_int* new_row_3 = w_ci; w_ci += A_.nnz();

    qp_int* w_qp = reinterpret_cast<qp_int*>(w_ci);
    qp_int* Pjc = w_qp; w_qp += H_.size2()+1;
    qp_int* Pir = w_qp; w_qp += H_.nnz();
    qp_int* Ajc = w_qp; w_qp += nx_ + A_.size2()+1;
    qp_int* Air = w_qp; w_qp += nx_ + A_.nnz();
    qp_int* Gjc = w_qp; w_qp += 2*(nx_ + A_.size2()+1);
    qp_int* Gir = w_qp; w_qp += 2*(nx_ + A_.nnz());

    uout() << "Data locations: " << std::endl;
    uout() << "lbx, ubx: " << lbx << ", " << ubx << std::endl;
    uout() << "lba, uba: " << lba << ", " << uba << std::endl;
    uout() << "P: " << Pjc << ", " << Pir << ", " << Ppr << std::endl;
    uout() << "A: " << Ajc << ", " << Air << ", " << Apr << std::endl;
    uout() << "G: " << Gjc << ", " << Gir << ", " << Gpr << std::endl;
    uout() << "c, h, b: " << c << ", " << h  << ", " << b << std::endl;
    uout() << "new_data: " << new_data << std::endl;
    uout() << "new_data_gA: " << new_data_gA_upper << ", " << new_data_gA_lower << std::endl;
    uout() << "a_prob_trans: " << a_prob_trans << std::endl;
    uout() << "iw: " << iw << std::endl;

    uout() << "lbx: " << std::vector<double>(lbx, lbx+nx_) << std::endl; 
    uout() << "ubx: " << std::vector<double>(ubx, ubx+nx_) << std::endl; 
    uout() << "lba: " << std::vector<double>(lba, lba+na_) << std::endl;
    uout() << "uba: " << std::vector<double>(uba, uba+na_) << std::endl;

    // indices
    casadi_int index_1, index_2, index_3;

    // QP var
    QP *qp_prob;

    // casadi_zero
    const casadi_int c_zero = 0;

    // Check A constraints
    casadi_int un_c_A = 0;
    casadi_int un_l_c_A = 0;
    casadi_int un_u_c_A = 0;
    index_1 = 0;
    index_2 = 0;
    index_3 = 0;
    for (casadi_int i = 0; i < na_; ++i) {
      if (lba[i] == uba[i]) {
        equality_constr_A[index_1++] = i;
      } else if (lba[i] == -inf && uba[i] == inf) {
        un_c_A++;
        unbounded_lower_constr_A[index_2++] = i;
        unbounded_upper_constr_A[index_3++] = i;
      } else if (lba[i] == -inf) {
        un_l_c_A++;
        unbounded_lower_constr_A[index_2++] = i;
      } else if (uba[i] == inf) {
        un_u_c_A++;
        unbounded_upper_constr_A[index_3++] = i;
      } 
    }

    uout() << "Equality constraints in A: " << std::vector<casadi_int>(equality_constr_A, equality_constr_A+index_1) << std::endl;
    uout() << "Unbounded lower A constraints: " << std::vector<casadi_int>(unbounded_lower_constr_A, unbounded_lower_constr_A+index_2) << std::endl;
    uout() << "Unbounded upper A constraints: " << std::vector<casadi_int>(unbounded_upper_constr_A, unbounded_upper_constr_A+index_3) << std::endl;

    // Define var for readability
    casadi_int eq_c_A = index_1;

    // Check x constraints
    casadi_int un_c_x = 0;
    casadi_int un_l_c_x = 0;
    casadi_int un_u_c_x = 0;
    index_1 = 0;
    index_2 = 0;
    index_3 = 0;
    for (casadi_int i = 0; i < nx_; ++i) {
      if (lbx[i] == ubx[i]) {
        equality_constr_x[index_1++] = i;
      } else if (lbx[i] == -inf && ubx[i] == inf) {
        un_c_x++;
        unbounded_lower_constr_x[index_2++] = i;
        unbounded_upper_constr_x[index_3++] = i;
      } else if (lbx[i] == -inf) {
        un_l_c_x++;
        unbounded_lower_constr_x[index_2++] = i;
      } else if (ubx[i] == inf) {
        un_u_c_x++;
        unbounded_upper_constr_x[index_3++] = i;
      }
    }

    uout() << "Equality constraints in x: " << std::vector<casadi_int>(equality_constr_x, equality_constr_x+index_1) << std::endl;
    uout() << "Unbounded lower x constraints: " << std::vector<casadi_int>(unbounded_lower_constr_x, unbounded_lower_constr_x+index_2) << std::endl;
    uout() << "Unbounded upper x constraints: " << std::vector<casadi_int>(unbounded_upper_constr_x, unbounded_upper_constr_x+index_3) << std::endl;

    // Define var for readability
    casadi_int eq_c_x = index_1;

    // Error if there are only equality constraints
    if (eq_c_x + un_c_x == nx_ && eq_c_A + un_c_A == na_) {
      casadi_error("qpSWIFT is not (yet) capable to solve qp's with only equality constraints...");
    }

    // Number of desiscion vars and constraints
    qp_int nc = nx_;
    qp_int mc = 2*(nx_+na_-eq_c_A-eq_c_x-un_c_x-un_c_A)-un_l_c_x-un_u_c_x-un_l_c_A-un_u_c_A; // 2 times for double inequalities
    qp_int pc = eq_c_A + eq_c_x;

    // P sparsity
    std::vector<qp_int> Pjc_vec = vector_static_cast<qp_int>(H_.get_colind());
    std::vector<qp_int> Pir_vec = vector_static_cast<qp_int>(H_.get_row());
    casadi_copy(Pjc_vec.data(), H_.size2()+1, Pjc);
    casadi_copy(Pir_vec.data(), H_.nnz(), Pir);

    uout() << "H/P sparsity: " << std::endl;
    H_.spy(uout());

    // P data
    uout() << "H/P data: " << std::vector<qp_real>(Ppr, Ppr+H_.nnz()) << std::endl;

    // A Sparsity, Ax part
    Sparsity A_trans = A_.T(); // Get transpose because we need to copy columns
    const double *a_prob = arg[CONIC_A];
    const casadi_int *A_trans_colind, *A_trans_row;
    Sparsity A_sp_a_trans;
    casadi_fill(new_data, 2*(nx_+A_.nnz()), 1.);

    double* a_prob_trans_temp = a_prob_trans;
    casadi_trans(a_prob, A_, a_prob_trans, A_trans, iw);
    if (eq_c_A > 0) {
      A_trans_colind = A_trans.colind();
      A_trans_row = A_trans.row();

      new_colind_1[0] = 0;
      index_1 = 0;
      for (casadi_int i = 0; i < A_.size1(); ++i) { // Loop over all the constraints
        if (equality_constr_A[index_1] == i) {
          casadi_copy(A_trans_row, *(A_trans_colind+1) - *A_trans_colind, new_row_1+new_colind_1[index_1]);
          casadi_copy(a_prob_trans_temp, *(A_trans_colind+1) - *A_trans_colind, new_data+new_colind_1[index_1]+eq_c_x);
          new_colind_1[index_1+1] = new_colind_1[index_1] + *(A_trans_colind+1) - *A_trans_colind;
          index_1++;
          if (index_1 == eq_c_A) break;
        } 

        A_trans_row += *(A_trans_colind+1) - *A_trans_colind;
        a_prob_trans_temp += *(A_trans_colind+1) - *A_trans_colind;
        A_trans_colind++;
      }
      A_sp_a_trans = Sparsity(nx_, eq_c_A, new_colind_1, new_row_1);
    } else {
      A_sp_a_trans = Sparsity(0,0);
    }
    
    // A Sparsity, x part
    Sparsity A_sp_trans;
    if (eq_c_x > 0) {
      A_sp_trans = Sparsity(nx_, eq_c_x);
      for (casadi_int i = 0; i < eq_c_x; ++i) A_sp_trans.add_nz(equality_constr_x[i], i);
    } else {
      A_sp_trans = Sparsity(0,0);
    }

    // A Sparsity
    Sparsity A_sp;
    if (eq_c_x > 0 || eq_c_A > 0) {
      A_sp_trans.appendColumns(A_sp_a_trans);
      A_sp = A_sp_trans.T();
      std::vector<qp_int> Ajc_vec = vector_static_cast<qp_int>(A_sp.get_colind());
      std::vector<qp_int> Air_vec = vector_static_cast<qp_int>(A_sp.get_row());
      casadi_copy(Ajc_vec.data(), A_sp.size2()+1, Ajc);
      casadi_copy(Air_vec.data(), A_sp.nnz(), Air);

      uout() << "equality constr sparsity: " << std::endl;
      A_sp.spy(uout());
    } else {
      Ajc = NULL;
      Air = NULL;
    }

    // A Data
    if (eq_c_A == 0 && eq_c_x == 0) {
      Apr = NULL;
    } else {
      casadi_trans(new_data, A_sp_trans, Apr, A_sp, iw);
      uout() << "equality constr data: " << std::vector<qp_real>(Apr, Apr+A_sp.nnz()) << std::endl;
    }

    // G sparsity: Ax upper bound part
    Sparsity G_sp_g_trans_upper;
    if (eq_c_A+un_c_A+un_u_c_A != na_) {
      a_prob = arg[CONIC_A];
      A_trans_colind = A_trans.colind();
      A_trans_row = A_trans.row();
      a_prob_trans_temp = a_prob_trans;

      new_colind_2[0] = 0;
      index_1 = 0;
      index_2 = 0;
      index_3 = 0;
      for (casadi_int i = 0; i < A_.size1(); ++i) { // Loop over all the constraints
        if (eq_c_A == 0 || equality_constr_A[index_1] != i) {
          if ((un_c_A == 0 && un_u_c_A == 0) || unbounded_upper_constr_A[index_2] != i) {
            casadi_copy(A_trans_row, *(A_trans_colind+1) - *A_trans_colind, new_row_2+new_colind_2[index_3]);
            casadi_copy(a_prob_trans_temp, *(A_trans_colind+1) - *A_trans_colind, new_data_gA_upper+new_colind_2[index_3]);
            new_colind_2[index_3+1] = new_colind_2[index_3] + *(A_trans_colind+1) - *A_trans_colind;
            index_3++;
          } else {
            index_2++;
          }  
        } else {
          index_1++;
        }

        A_trans_row += *(A_trans_colind+1) - *A_trans_colind;
        a_prob_trans_temp += *(A_trans_colind+1) - *A_trans_colind;
        A_trans_colind++;
      }
      G_sp_g_trans_upper = Sparsity(nx_, na_-eq_c_A-un_c_A-un_u_c_A, new_colind_2, new_row_2);
    } else {
      G_sp_g_trans_upper = Sparsity(0,0);
    }

    // G sparsity: Ax lower bound part
    Sparsity G_sp_g_trans_lower;
    if (eq_c_A+un_c_A+un_l_c_A != na_) {
      a_prob = arg[CONIC_A];
      A_trans_colind = A_trans.colind();
      A_trans_row = A_trans.row();
      a_prob_trans_temp = a_prob_trans;

      new_colind_3[0] = 0;
      index_1 = 0;
      index_2 = 0;
      index_3 = 0;
      for (casadi_int i = 0; i < A_.size1(); ++i) { // Loop over all the constraints
        if (eq_c_A == 0 || equality_constr_A[index_1] != i) {
          if ((un_c_A == 0 && un_l_c_A == 0) || unbounded_lower_constr_A[index_2] != i) {
            casadi_copy(A_trans_row, *(A_trans_colind+1) - *A_trans_colind, new_row_3+new_colind_3[index_3]);
            casadi_copy(a_prob_trans_temp, *(A_trans_colind+1) - *A_trans_colind, new_data_gA_lower+new_colind_3[index_3]);
            new_colind_3[index_3+1] = new_colind_3[index_3] + *(A_trans_colind+1) - *A_trans_colind;
            index_3++;
          } else {
            index_2++;
          }        
        } else {
          index_1++;
        }

        A_trans_row += *(A_trans_colind+1) - *A_trans_colind;
        a_prob_trans_temp += *(A_trans_colind+1) - *A_trans_colind;
        A_trans_colind++;
      }
      G_sp_g_trans_lower = Sparsity(nx_, na_-eq_c_A-un_c_A-un_l_c_A, new_colind_3, new_row_3);
    } else {
      G_sp_g_trans_lower = Sparsity(0,0);
    }

    // G sparsity, x upper bound part
    Sparsity G_sp_trans_upper;
    if (eq_c_x+un_c_x+un_u_c_x != nx_) {
      Sparsity G_sp_x_trans = Sparsity(nx_, nx_-eq_c_x-un_c_x-un_u_c_x);
      index_1 = 0;
      index_2 = 0;
      index_3 = 0;
      for (casadi_int i = 0; i < nx_; ++i) {
        if (eq_c_x == 0 || i != equality_constr_x[index_1]) {
          if ((un_c_x == 0 && un_u_c_x == 0) || i != unbounded_upper_constr_x[index_2]) {
            G_sp_x_trans.add_nz(i, index_3);
            index_3++;
          } else {
            index_2++;
          }
        } else {
          index_1++;
        }
      }
      G_sp_trans_upper = Sparsity(G_sp_x_trans);
    } else {
      G_sp_trans_upper = Sparsity(0,0);
    }

    // G sparsity: x lower bound part
    Sparsity G_sp_trans_lower;
    if (eq_c_x+un_c_x+un_l_c_x != nx_) {
      Sparsity G_sp_x_trans = Sparsity(nx_, nx_-eq_c_x-un_c_x-un_l_c_x);
      index_1 = 0;
      index_2 = 0;
      index_3 = 0;
      for (casadi_int i = 0; i < nx_; ++i) {
        if (eq_c_x == 0 || i != equality_constr_x[index_1]) {
          if ((un_c_x == 0 && un_l_c_x == 0) || i != unbounded_lower_constr_x[index_2]) {
            G_sp_x_trans.add_nz(i, index_3);
            index_3++;
          } else {
            index_2++;
          }
        } else {
          index_1++;
        }
      }
      G_sp_trans_lower = Sparsity(G_sp_x_trans);
    } else {
      G_sp_trans_lower = Sparsity(0,0);
    }

    // G sparsity
    Sparsity G_sp, G_sp_T;
    if (eq_c_A+un_c_A != na_ || eq_c_x+un_c_x != nx_) {
      G_sp_T = Sparsity(G_sp_trans_upper);
      G_sp_T.appendColumns(G_sp_trans_lower);
      G_sp_T.appendColumns(G_sp_g_trans_upper);
      G_sp_T.appendColumns(G_sp_g_trans_lower);
      G_sp = G_sp_T.T();

      std::vector<qp_int> Gjc_vec = vector_static_cast<qp_int>(G_sp.get_colind());
      std::vector<qp_int> Gir_vec = vector_static_cast<qp_int>(G_sp.get_row());
      casadi_copy(Gjc_vec.data(), G_sp.size2()+1, Gjc);
      casadi_copy(Gir_vec.data(), G_sp.nnz(), Gir);

      uout() << "A/G sparsity: " << std::endl;
      G_sp.spy(uout());
    } else {
      Gjc = NULL;
      Gir = NULL;
    }
    
    // G data
    if (eq_c_A+un_c_A != na_ || eq_c_x+un_c_x != nx_) {
      casadi_fill(new_data, G_sp_trans_upper.nnz(), 1.);
      casadi_fill(new_data+G_sp_trans_upper.nnz(), G_sp_trans_lower.nnz(), -1.);
      casadi_copy(new_data_gA_upper, G_sp_g_trans_upper.nnz(), new_data+G_sp_trans_upper.nnz()+G_sp_trans_lower.nnz());
      casadi_clear(new_data+G_sp_trans_upper.nnz()+G_sp_trans_lower.nnz()+G_sp_g_trans_upper.nnz(), G_sp_g_trans_lower.nnz());
      casadi_axpy(G_sp_g_trans_lower.nnz(), -1., new_data_gA_lower, new_data+G_sp_trans_upper.nnz()+G_sp_trans_lower.nnz()+G_sp_g_trans_upper.nnz());
      uout() << "new_data: " << new_data << std::vector<qp_real>(new_data, new_data+G_sp.nnz()) << std::endl;
      uout() << "Gpr: " << Gpr << std::endl;
      casadi_trans(new_data, G_sp_T, Gpr, G_sp, iw);
      uout() << "A/G data: " << std::vector<qp_real>(Gpr, Gpr+G_sp.nnz()) << std::endl;
    } else {
      Gpr = NULL;
    }

    // c data
    if (arg[CONIC_G]) {
      casadi_copy(arg[CONIC_G], nx_, c);
      uout() << "g/C data: " << std::vector<qp_real>(c, c+nx_) << std::endl;
    } else {
      c = NULL;
    }

    // h Data
    if (eq_c_A+un_c_A != na_ || eq_c_x+un_c_x != nx_) {
      casadi_int h_index = 0;
      
      // x upper part
      index_1 = 0;
      index_2 = 0;
      for (casadi_int i = 0; i < nx_; ++i) {
        if (eq_c_x == 0 || i != equality_constr_x[index_1]) {
          if ((un_c_x == 0 && un_u_c_x == 0) || i != unbounded_upper_constr_x[index_2]) {
            h[h_index] = ubx[i];
            h_index++;
          } else {
            index_2++;
          }
        } else {
          index_1++;
        }
      }

      // x lower part
      index_1 = 0;
      index_2 = 0;
      for (casadi_int i = 0; i < nx_; ++i) {
        if (eq_c_x == 0 || i != equality_constr_x[index_1]) {
          if ((un_c_x == 0 && un_l_c_x == 0) || i != unbounded_lower_constr_x[index_2]) {
            h[h_index] = -lbx[i];
            h_index++;
          } else {
            index_2++;
          }
        } else {
          index_1++;
        }
      }
      
      // A upper part
      index_1 = 0;
      index_2 = 0;
      for (casadi_int i = 0; i < na_; ++i) {
        if (eq_c_A == 0 || i != equality_constr_A[index_1]) {
          if ((un_c_A == 0 && un_u_c_A == 0) || i != unbounded_upper_constr_A[index_2]) {
            h[h_index] = uba[i];
            h_index++;
          } else {
            index_2++;
          }
        } else {
          index_1++;
        }
      }

      // A lower part
      index_1 = 0;
      index_2 = 0;
      for (casadi_int i = 0; i < na_; ++i) {
        if (eq_c_A == 0 || i != equality_constr_A[index_1]) {
          if ((un_c_A == 0 && un_l_c_A == 0) || i != unbounded_lower_constr_A[index_2]) {
            h[h_index] = -lba[i];
            h_index++;
          } else {
            index_2++;
          }
        } else {
          index_1++;
        }
      }

      uout() << "bnds/h data: " << std::vector<double>(h, h+h_index) << std::endl;
    } else {
      h = NULL;
    }

    // b Data
    if (eq_c_A > 0 || eq_c_x > 0) {
      casadi_int b_index = 0;
      if (eq_c_x > 0) {
        index_1 = 0;
        for (casadi_int i = 0; i < nx_; ++i) {
          if (i == equality_constr_x[index_1]) {
              b[b_index] = ubx[i];
              b_index++;
              index_1++;
          }
        }
      }
      
      if (eq_c_A > 0) {
        index_1 = 0;
        for (casadi_int i = 0; i < na_; ++i) {
          if (i == equality_constr_A[index_1]) {
              b[b_index] = uba[i];
              b_index++;
              index_1++;
          }
        }
      }
    
      uout() << "equality bnds: " << std::vector<double>(b, b+b_index) << std::endl;
    } else {
      b = NULL;
    }

    // Get solver
    uout() << "Problem sizes: " << nc << ", " << mc << ", " << pc << std::endl;
    uout() << "Data locations: " << std::endl;
    uout() << "P: " << Pjc << ", " << Pir << ", " << Ppr << std::endl;
    uout() << "A: " << Ajc << ", " << Air << ", " << Apr << std::endl;
    uout() << "G: " << Gjc << ", " << Gir << ", " << Gpr << std::endl;
    uout() << "c, h, b: " << c << ", " << h  << ", " << b << std::endl;
    qp_prob = QP_SETUP(nc, mc, pc, Pjc, Pir, Ppr, Ajc, Air, Apr, Gjc, Gir, Gpr, c, h, b, 
                    0, NULL);

    // Change solver settings
    if (maxit_ != 0) qp_prob->options->maxit = maxit_;
    if (reltol_ != 0.) qp_prob->options->reltol = reltol_;
    if (abstol_ != 0.) qp_prob->options->abstol = abstol_;
    if (sigma_ != 0.) qp_prob->options->sigma = sigma_;
    if (verbose_ != 0) qp_prob->options->verbose = verbose_;

    // Solve QP
    qp_int exit_code = QP_SOLVE(qp_prob);
    if (exit_code == QP_OPTIMAL) {
      uout() << "qpswift succeeded" << std::endl;
      m->success = true;
      m->unified_return_status = SOLVER_RET_SUCCESS;
    } else if (exit_code == QP_FATAL || exit_code == QP_KKTFAIL || exit_code == QP_MAXIT) {
      uout() << "qpswift failed due to " << exit_code << std::endl;
      m->success = false;
      m->unified_return_status = SOLVER_RET_INFEASIBLE;
      return 1;
    } 

    uout() << "Copying x" << std::endl;

    // Copy output x
    casadi_copy(qp_prob->x, nx_, res[CONIC_X]);

    uout() << "Copying lam_x" << std::endl;

    // Copy output lam_x
    double* lam_x = res[CONIC_LAM_X];
    double* z_out = qp_prob->z;
    double* z_out_dual = qp_prob->z +nx_-eq_c_x-un_c_x-un_u_c_x;
    double* y_out = qp_prob->y;

    index_1 = 0;
    index_2 = 0;
    index_3 = 0;
    for (casadi_int i = 0; i < nx_; ++i) {
      if (eq_c_x > 0 && i == equality_constr_x[index_1]) {
        *lam_x++ = *y_out++;
        index_1++;
      } else {
        *lam_x = 0;
        if ((un_c_x == 0 && un_u_c_x == 0) || i != unbounded_upper_constr_x[index_2]) {
          *lam_x += *z_out++;
        } else {
          index_2++;
        }
        if ((un_c_x == 0 && un_l_c_x == 0) || i != unbounded_lower_constr_x[index_3]) {
          *lam_x -= *z_out_dual++;
        } else {
          index_3++;
        }
        lam_x++;
      }
    }
    
    uout() << "Copying lam_a" << std::endl;

    // Copy output lam_a
    double* lam_a = res[CONIC_LAM_A];
    z_out = z_out_dual;
    z_out_dual = z_out+na_-eq_c_A-un_c_A-un_u_c_A;

    index_1 = 0;
    index_2 = 0;
    index_3 = 0;
    for (casadi_int i = 0; i < na_; ++i) {
      if (eq_c_A > 0 && i == equality_constr_A[index_1]) {
        *lam_a++ = *y_out++;
        index_1++;
      } else {
        *lam_a = 0;
        if ((un_c_A == 0 && un_u_c_A == 0) || i != unbounded_upper_constr_A[index_2]) {
          *lam_a += *z_out++;
        } else {
          index_2++;
        }
        if ((un_c_A == 0 && un_l_c_A == 0) || i != unbounded_lower_constr_A[index_3]) {
          *lam_a -= *z_out_dual++;
        } else {
          index_3++;
        } 
        lam_a++;
      }
      
    }

    uout() << "Copying cost" << std::endl;

    // Copy cost
    if (res[CONIC_COST]) *res[CONIC_COST] = qp_prob->stats->fval;

    // Copy stats
    m->tsetup = qp_prob->stats->tsetup;
    m->tsolve = qp_prob->stats->tsolve;
    m->kkt_time = qp_prob->stats->kkt_time;
    m->ldl_numeric = qp_prob->stats->ldl_numeric;
    m->iter_count = static_cast<int>(qp_prob->stats->IterationCount);

    uout() << "Entering cleanup" << std::endl;

    // Cleanup
    QP_CLEANUP(qp_prob);

    return 0;
  }

  void QpswiftInterface::codegen_init_mem(CodeGenerator& g) const {
    g << "return 0;\n";
  }

  void QpswiftInterface::codegen_body(CodeGenerator& g) const {
    g.add_include("qpSWIFT/qpSWIFT.h");
    g.add_auxiliary(CodeGenerator::AUX_INF);

    g.local("work", "QPSWIFTWorkspace", "*");
    g.init_local("work", codegen_mem(g));

    g.comment("Set memory locations and copy some vectors");
    g << "double * lbx = w; w += " << nx_ << ";\n";
    g.copy_default(g.arg(CONIC_LBX), nx_, "lbx", "-casadi_inf", false);
    g << "double * ubx = w; w += " << nx_ << ";\n";
    g.copy_default(g.arg(CONIC_UBX), nx_, "ubx", "casadi_inf", false);
    g << "double * lba = w; w += " << na_ << ";\n";
    g.copy_default(g.arg(CONIC_LBX), nx_, "lba", "-casadi_inf", false);
    g << "double * uba = w; w += " << na_ << ";\n";
    g.copy_default(g.arg(CONIC_UBX), nx_, "uba", "casadi_inf", false);
    g << "double* Ppr = w; w += " << H_.nnz() << ";\n";
    g.copy(g.arg(CONIC_H), H_.nnz(), "Ppr");
    g << "double* Apr=w; w += " << A_.nnz() << ";\n";
    g << "double* Gpr=w; w += " << A_.nnz() << ";\n";
    g << "double* c=w; w += " << nx_ << ";\n";
    g << "double* h=w; w += " << 2*(nx_+na_) << ";\n";
    g << "double* b=w; w += " << nx_+na_ << ";\n";
    g << "double* new_data=w; w += " << 2*(nx_+A_.nnz()) << ";\n";
    g << "double* a_prob_trans=w; w += " << A_.nnz() << ";\n";
    g << "double* new_data_gA_upper=w; w += " << A_.nnz() << ";\n";
    g << "double* new_data_gA_lower=w; w += " << A_.nnz() << ";\n";

    g << "casadi_int* w_ci = (qp_int*)w;\n";
    g << "casadi_int* equality_constr_A = w_ci; w_ci += " << na_ << ";\n";
    g << "casadi_int* unbounded_lower_constr_A = w_ci; w_ci += " << na_ << ";\n";
    g << "casadi_int* unbounded_upper_constr_A = w_ci; w_ci += " << na_ << ";\n";
    g << "casadi_int* equality_constr_x = w_ci; w_ci += " << nx_ << ";\n";
    g << "casadi_int* unbounded_lower_constr_x = w_ci; w_ci += " << nx_ << ";\n";
    g << "casadi_int* unbounded_upper_constr_x = w_ci; w_ci += " << nx_ << ";\n";

    g << "casadi_int* new_colind_1 = w_ci; w_ci += " << nx_+1+na_+1 << ";\n";
    g << "casadi_int* new_colind_2 = w_ci; w_ci += " << 2*(nx_+1)+na_+1 << ";\n";
    g << "casadi_int* new_colind_3 = w_ci; w_ci += " << na_+1 << ";\n";
    g << "casadi_int* new_row_1 = w_ci; w_ci += " << nx_+A_.nnz() << ";\n";
    g << "casadi_int* new_row_2 = w_ci; w_ci += " << 2*nx_+A_.nnz() << ";\n";
    g << "casadi_int* new_row_3 = w_ci; w_ci += " << A_.nnz() << ";\n";

    g << "qp_int* w_qp = (qp_int*)w_ci;\n";
    g << "qp_int* Pjc = w_qp; w_qp += " << H_.size2()+1 << ";\n";
    g << "qp_int* Pir = w_qp; w_qp += " << H_.nnz() << ";\n";
    g << "qp_int* Ajc = w_qp; w_qp += " << nx_ + A_.size2()+1 << ";\n";
    g << "qp_int* Air = w_qp; w_qp += " << nx_ + A_.nnz() << ";\n";
    g << "qp_int* Gjc = w_qp; w_qp += " << 2*(nx_ + A_.size2()+1) << ";\n";
    g << "qp_int* Gir = w_qp; w_qp += " << 2*(nx_ + A_.nnz()) << ";\n";

    g.comment("indices");
    g << "casadi_int index_1, index_2, index_3;\n";

    g.comment("QP var");
    g << "QP *qp_prob;\n";

    g.comment("casadi_zero");
    g << "const casadi_int c_zero = 0;\n";

    g.comment("Check A constraints");
    g << "casadi_int un_c_A = 0;\n";
    g << "casadi_int un_l_c_A = 0;\n";
    g << "casadi_int un_h_c_A = 0;\n";
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "index_3 = 0;\n";
    g << "for (casadi_int i = 0; i < " << na_ << "; ++i) {\n";
    g << "if (lba[i] == uba[i]) {\n";
    g << "equality_constr_A[index_1++] = i;\n";
    g << "} else if (lba[i] == -casadi_inf && uba[i] == casadi_inf) {\n";
    g << "un_c_A++;\n";
    g << "unbounded_lower_constr_A[index_2++] = i;\n";
    g << "unbounded_upper_constr_A[index_3++] = i;\n";
    g << "} else if (lba[i] == -casadi_inf) {\n";
    g << "un_l_c_A++;\n";
    g << "unbounded_lower_constr_A[index_2++] = i;\n";
    g << "} else if (uba[i] == casadi_inf) {\n";
    g << "un_u_c_A++;\n";
    g << "unbounded_upper_cosntr_A[index_3++] = i;\n";
    g << "}\n";
    g << "}\n";

    g.comment("Define var for readability");
    g << "casadi_int eq_c_A = index_1;\n";

    g.comment("Check x constraints");
    g << "casadi_int un_c_x = 0;\n";
    g << "casadi_int un_l_c_x = 0;\n";
    g << "casadi_int un_h_c_x = 0;\n";
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "index_3 = 0;\n";
    g << "for (casadi_int i = 0; i < " << nx_ << "; ++i) {\n";
    g << "if (lbx[i] == ubx[i]) {\n";
    g << "equality_constr_x[index_1++] = i;\n";
    g << "} else if (lbx[i] == -casadi_inf && ubx[i] == casadi_inf) {\n";
    g << "un_c_x++;\n";
    g << "unbounded_lower_constr_x[index_2++] = i;\n";
    g << "unbounded_upper_constr_x[index_3++] = i;\n";
    g << "} else if (lbx[i] == -casadi_inf) {\n";
    g << "un_l_c_x++;\n";
    g << "unbounded_lower_constr_x[index_2++] = i;\n";
    g << "} else if (ubx[i] == casadi_inf) {\n";
    g << "un_u_c_x++;\n";
    g << "unbounded_upper_cosntr_x[index_3++] = i;\n";
    g << "}\n";
    g << "}\n";

    g.comment("Define var for readability");
    g << "casadi_int eq_c_x = index_1;\n";

    g.comment("Error if there are only equality constraints");
    g << "if (eq_c_x + un_c_x == " << nx_ << " && eq_c_A + un_c_A == " << na_ << ") {\n";
    g << "return -1;\n";
    g << "}\n";

    g.comment("Number of desiscion vars and constraints");
    g << "qp_int nc = " << nx_ << ";\n";
    g << "qp_int mc = 2*(" << nx_+na_ << "-eq_c_A-eq_c_x-un_c_x-un_c_A)-un_l_c_x-un_u_c_x-un_l_c_A-un_u_c_A;\n";
    g << "qp_int pc = eq_c_A + eq_c_x";

    g.comment("P sparsity");
    g.constant_copy("Pjc", H_.get_colind(), "qp_int");
    g.constant_copy("Pir", H_.get_row(), "qp_int");

    g.comment("A sparsity, x part");
    g << "new_colind_1[0] = 0";
    g << "if (eq_c_x > 0) {\n";
    g << "for (casadi_int i = 0; i < eq_c_x; ++i) {\n";
    g << "new_colind_1[i+1] = new_colind_1[i] + 1;\n";
    g << "new_row_1[i] = equality_constr_x[i];\n";
    g << "}\n";
    g << "}\n";
    g << "index_1 = i;\n";

    g.comment("A sparsity, Ax part (here we use temp variables Ajc and Air)");
    Sparsity A_trans = A_.T();
    g << "const double *a_prob = " << g.arg(CONIC_A) << ";\n";
    g.constant_copy("Ajc", A_trans.get_colind(), "qp_int");
    g.constant_copy("Air", A_trans.get_row(), "qp_int");
    g << "const qp_int *A_trans_colind, *A_trans_row;\n";
    g << g.fill("new_data", 2*(nx_+A_.nnz()), "1.") << "\n";
    
    g << "double* a_prob_trans_temp = a_prob_trans;\n";
    g << g.trans("a_prob", A_, "a_prob_trans", A_trans, "&iw") << "\n";
    g << "if (eq_c_A > 0) {\n";
    g << "A_trans_colind = Ajc;\n";
    g << "A_trans_row = Air;\n";
    
    g << "for (casadi_int i = 0; i < " << A_.size1() << "; ++i) {\n";
    g << "if (equality_constr_A[index_1] == i) {\n";
    g << "casadi_copy(A_trans_row, *(A_trans_colind+1) - *A_trans_colind, new_row_1+new_colind_1[index_1]);\n";
    g << "casadi_copy(a_prob_trans_temp, *(A_trans_colind+1) - *A_trans_colind, new_data+new_colind_1[index_1]+eq_c_x);\n";
    g << "new_colind_1[index_1+1] = new_colind_1[index_1] + *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "index_1++;\n";
    g << "if (index_1 == eq_c_A) break;\n";
    g << "}\n";

    g << "A_trans_row += *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "a_prob_trans_temp += *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "A_trans_colind++;\n";
    g << "}\n";
    g << "}\n";

    g.comment("A Sparsity");
    g << "if (eq_c_x > 0 || eq_c_A > 0) {\n";
    codegen_transpose_sparsity(g, str(nx_), "eq_c_x+eq_c_A", "Air", "Ajc");
    g << "} else {\n";
    g << "Ajc = NULL;\n";
    g << "Air = NULL;\n";
    g << "}\n";

    g.comment("A data");
    g << "if (eq_c_A == 0 && eq_c_x == 0) {\n";
    g << "Apr = NULL;\n";
    g << "} else {\n";
    codegen_transpose_data(g, str(nx_), "eq_c_x+eq_c_A", "Air", "Ajc", "new_data", "Apr");
    g << "}\n";

    g.comment("G sparsity: x upper bound part");
    g << "new_colind_2[0] = 0;\n";
    g << "if (eq_c_x+un_c_x+un_u_c_x != " << nx_ << ") {\n";
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "index_3 = 0;\n";
    g << "for (casadi_int i = 0; i < " << nx_ << "; ++i) {\n";
    g << "if (eq_c_x == 0 || i != equality_constr_x[index_1]) {\n";
    g << "if ((un_c_x == 0 && un_u_c_x == 0) || i != unbounded_upper_constr_x[index_2]) {\n";
    g << "new_colind_2[index_3+1] = Ajc[index_3] + 1;\n";
    g << "new_row_2[i] = i;\n";
    g << "index_3++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";
    g << "}\n";

    g.comment("G sparsity: x lower bound part");
    g << "new_colind_2[0] = 0;\n";
    g << "if (eq_c_x+un_c_x+un_u_c_x != " << nx_ << ") {\n";
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "for (casadi_int i = 0; i < " << nx_ << "; ++i) {\n";
    g << "if (eq_c_x == 0 || i != equality_constr_x[index_1]) {\n";
    g << "if ((un_c_x == 0 && un_u_c_x == 0) || i != unbounded_lower_constr_x[index_2]) {\n";
    g << "new_colind_2[index_3+1] = Ajc[index_3] + 1;\n";
    g << "new_row_2[i] = i;\n";
    g << "index_3++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";
    g << "}\n";

    g.comment("G sparsity: Ax upper bound part");
    g << "if (eq_c_A+un_c_A+un_u_c_A != na_) {\n";
    g.constant_copy("Gjc", A_trans.get_colind(), "qp_int");
    g.constant_copy("Gir", A_trans.get_row(), "qp_int");
    g << "a_prob = " << g.arg(CONIC_A) << ";\n";
    g << "A_trans_colind = Gjc;\n";
    g << "A_trans_row = Gir;\n";
    g << "a_prob_trans_temp = a_prob_trans;\n";

    g << "new_colind_2[0] = 0;\n";
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "for (casadi_int i = 0; i < " << A_.size1() << "; ++i) {\n";
    g << "if (eq_c_A == 0 || equality_constr_A[index_1] != i) {\n";
    g << "if ((un_c_A == 0 && un_u_c_A == 0) || unbounded_upper_constr_A[index_2] != i) {\n";
    g << "casadi_copy(A_trans_row, *(A_trans_colind+1) - *A_trans_colind, new_row_2+new_colind_2[index_3]);\n";
    g << "casadi_copy(a_prob_trans_temp, *(A_trans_colind+1) - *A_trans_colind, new_data_gA_upper+new_colind_2[index_3]);\n";
    g << "new_colind_2[index_3+1] = new_colind_2[index_3] + *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "index_3++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";

    g << "A_trans_row += *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "a_prob_trans_temp += *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "A_trans_colind++;";
    g << "}\n";
    g << "}\n";

    g.comment("G sparsity: Ax lower bound part");
    g << "if (eq_c_A+un_c_A+un_l_c_A != na_) {\n";
    g << "a_prob = " << g.arg(CONIC_A) << ";\n";
    g << "A_trans_colind = Gjc;\n";
    g << "A_trans_row = Gir;\n";
    g << "a_prob_trans_temp = a_prob_trans;\n";

    g << "new_colind_3[0] = 0;\n";
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "index_3 = 0;\n";
    g << "for (casadi_int i = 0; i < " << A_.size1() << "; ++i) {\n";
    g << "if (eq_c_A == 0 || equality_constr_A[index_1] != i) {\n";
    g << "if ((un_c_A == 0 && un_l_c_A == 0) || unbounded_lower_constr_A[index_2] != i) {\n";
    g << "casadi_copy(A_trans_row, *(A_trans_colind+1) - *A_trans_colind, new_row_3+new_colind_3[index_3]);\n";
    g << "casadi_copy(a_prob_trans_temp, *(A_trans_colind+1) - *A_trans_colind, new_data_gA_lower+new_colind_3[index_3]);\n";
    g << "new_colind_3[index_3+1] = new_colind_3[index_3] + *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "index_3++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";

    g << "A_trans_row += *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "a_prob_trans_temp += *(A_trans_colind+1) - *A_trans_colind;\n";
    g << "A_trans_colind++;\n";
    g << "}\n";
    g << "}\n";

    g.comment("G sparsity");
    g << "if (eq_c_A+un_c_A != " << na_ << " || eq_c_x+un_c_x != " << nx_ << ") {\n";
    // TODO(@KobeBergmans)
    g << "} else {\n";
    g << "Gjc = NULL;\n";
    g << "}\n";

    g.comment("G data");
    g << "if (eq_c_A+un_c_A != " << na_ << " || eq_c_x+un_c_x != " << nx_ << ") {\n";
    // TODO(@KobeBergmans)
    g << "} else {\n";
    g << "Gpr = NULL;\n";
    g << "}\n";

    g.comment("c data");
    g << "if (" << g.arg(CONIC_G) << ") {\n";
    g << g.copy(g.arg(CONIC_G), nx_, "c") << "\n";
    g << "} else {\n";
    g << "c = NULL;\n";
    g << "}\n";

    g.comment("h data");
    g << "if (eq_c_A+un_c_A != " << na_ << " || eq_c_x+un_c_x != " << nx_ << ") {\n";
    g << "casadi_int h_index = 0;\n";
    
    g.comment("x upper part");
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "for (casadi_int i = 0; i < " << nx_ << "; ++i) {\n";
    g << "if (eq_c_x == 0 || i != equality_constr_x[index_1]) {\n";
    g << "if ((un_c_x == 0 && un_u_c_x == 0) || i != unbounded_upper_constr_x[index_2]) {\n";
    g << "h[h_index] = ubx[i];\n";
    g << "h_index++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";

    g.comment("x lower part");
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "for (casadi_int i = 0; i < " << nx_ << "; ++i) {\n";
    g << "if (eq_c_x == 0 || i != equality_constr_x[index_1]) {\n";
    g << "if ((un_c_x == 0 && un_l_c_x == 0) || i != unbounded_lower_constr_x[index_2]) {\n";
    g << "h[h_index] = -lbx[i];\n";
    g << "h_index++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";

    g.comment("A upper part");
    g << "index_1 = 0;\n";
    g << "index-2 = 0;\n";
    g << "for (casadi_int i = 0; i < " << na_ << "; ++i) {\n";
    g << "if (eq_c_A == 0 || i != equality_constr_A[index_1]) {\n";
    g << "if ((un_c_A == 0 && un_u_c_A == 0) || i != unbounded_upper_constr_A[index_2]) {\n";
    g << "h[h_index] = uba[i];\n";
    g << "h_index++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";

    g.comment("A lower part");
    g << "index_1 = 0;\n";
    g << "index_2 = 0;\n";
    g << "for (casadi_int i = 0; i < " << na_ << "; ++i) {\n";
    g << "if (eq_c_A == 0 || i != equality_constr_A[index_1]) {\n";
    g << "if ((un_c_A == 0 && un_l_c_A == 0) || i != unbounded_lower_constr_A[index_2]) {\n";
    g << "h[h_index] = -lba[i];\n";
    g << "h_index++;\n";
    g << "} else {\n";
    g << "index_2++;\n";
    g << "}\n";
    g << "} else {\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";

    g << "} else {\n";
    g << "h = NULL;\n";
    g << "}\n";

    g.comment("b Data");
    g << "if (eq_c_A > 0 || eq_c_x > 0) {\n";
    g << "casadi_int b_index = 0;\n";
    g << "if (eq_c_x > 0) {\n";
    g << "index_1 = 0;\n";
    g << "for (casadi_int i = 0; i < " << nx_ << "; ++i) {\n";
    g << "if (i == equality_constr_x[index_1]) {\n";
    g << "b[b_index] = ubx[i];\n";
    g << "b_index++;\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";
    g << "}\n";

    g << "if (eq_c_A > 0) {\n";
    g << "index_1 = 0;\n";
    g << "for (casadi_int i = 0; i < " << na_ << "; ++i) {\n";
    g << "if (i == equality_constr_A[index_1]) {\n";
    g << "b[b_index] = uba[i];\n";
    g << "b_index++;\n";
    g << "index_1++;\n";
    g << "}\n";
    g << "}\n";
    g << "}\n";

    g << "} else {\n";
    g << "b = NULL;\n";
    g << "}\n";

    g.comment("Get solver");
    g << "qp_prob = QP_SETUP(nc, mc, pc, Pjc, Pir, Ppr, Ajc, Air, Apr, Gjc, Gir, Gpr, c, h, b, 0, NULL);\n";

    g.comment("Change solver settings");
    if (maxit_ != 0) g << "qp_prob.options.maxit = " << maxit_ << ";\n";
    if (reltol_ != 0.) g << "qp_prob.options.reltol = " << reltol_ << ";\n";
    if (abstol_ != 0.) g << "qp_prob.options.abstol = " << abstol_ << ";\n";
    if (sigma_ != 0.) g << "qp_prob.options.sigma = " << sigma_ << ";\n";
    if (verbose_ != 0) g << "qp_prob.options.verbose = " << verbose_ << ";\n";

    g.comment("Solve QP");
    // TODO(@KobeBergmans)
  }

  void QpswiftInterface::codegen_transpose_sparsity(CodeGenerator& g, std::string nrow, std::string ncol,
                                           std::string row, std::string col) const {
    // TODO(@KobeBergmans): see sparsity_internal::triplet
    //                      Results should also end up in row and col vectors
  }

  void QpswiftInterface::codegen_transpose_data(CodeGenerator& g, std::string nrow, std::string ncol,
                                           std::string row, std::string col, std::string data, 
                                           std::string new_data) const {
    // TODO(@KobeBergmans)
  }

  Dict QpswiftInterface::get_stats(void* mem) const {
    Dict stats = Conic::get_stats(mem);
    for (auto&& op : stats_) {
      stats[op.first] = op.second;
    }
    
    return stats;
  }

  QpswiftMemory::QpswiftMemory() {
  }

  QpswiftMemory::~QpswiftMemory() {
  }

  QpswiftInterface::QpswiftInterface(DeserializingStream& s) : Conic(s) {
    s.version("QpswiftInterface", 1);
    s.unpack("QpswiftInterface::nnzHupp", nnzHupp_);
    s.unpack("QpswiftInterface::nnzA", nnzA_);
    s.unpack("QpswiftInterface::warm_start_primal", warm_start_primal_);
    s.unpack("QpswiftInterface::warm_start_dual", warm_start_dual_);
    s.unpack("QpswiftInterface::stats", stats_);
    // s.unpack("QpswiftInterface::settings::rho", settings_.rho);
    // s.unpack("QpswiftInterface::settings::sigma", settings_.sigma);
    // s.unpack("QpswiftInterface::settings::scaling", settings_.scaling);
    // s.unpack("QpswiftInterface::settings::adaptive_rho", settings_.adaptive_rho);
    // s.unpack("QpswiftInterface::settings::adaptive_rho_interval", settings_.adaptive_rho_interval);
    // s.unpack("QpswiftInterface::settings::adaptive_rho_tolerance", settings_.adaptive_rho_tolerance);
    //s.unpack("QpswiftInterface::settings::adaptive_rho_fraction", settings_.adaptive_rho_fraction);
    // s.unpack("QpswiftInterface::settings::max_iter", settings_.max_iter);
    // s.unpack("QpswiftInterface::settings::eps_abs", settings_.eps_abs);
    // s.unpack("QpswiftInterface::settings::eps_rel", settings_.eps_rel);
    // s.unpack("QpswiftInterface::settings::eps_prim_inf", settings_.eps_prim_inf);
    // s.unpack("QpswiftInterface::settings::eps_dual_inf", settings_.eps_dual_inf);
    // s.unpack("QpswiftInterface::settings::alpha", settings_.alpha);
    // s.unpack("QpswiftInterface::settings::delta", settings_.delta);
    // s.unpack("QpswiftInterface::settings::polish", settings_.polish);
    // s.unpack("QpswiftInterface::settings::polish_refine_iter", settings_.polish_refine_iter);
    // s.unpack("QpswiftInterface::settings::verbose", settings_.verbose);
    // s.unpack("QpswiftInterface::settings::scaled_termination", settings_.scaled_termination);
    // s.unpack("QpswiftInterface::settings::check_termination", settings_.check_termination);
    // s.unpack("QpswiftInterface::settings::warm_start", settings_.warm_start);
    //s.unpack("QpswiftInterface::settings::time_limit", settings_.time_limit);
  }

  void QpswiftInterface::serialize_body(SerializingStream &s) const {
    Conic::serialize_body(s);
    s.version("QpswiftInterface", 1);
    s.pack("QpswiftInterface::nnzHupp", nnzHupp_);
    s.pack("QpswiftInterface::nnzA", nnzA_);
    s.pack("QpswiftInterface::warm_start_primal", warm_start_primal_);
    s.pack("QpswiftInterface::warm_start_dual", warm_start_dual_);
    // s.pack("QpswiftInterface::settings::rho", settings_.rho);
    // s.pack("QpswiftInterface::settings::sigma", settings_.sigma);
    // s.pack("QpswiftInterface::settings::scaling", settings_.scaling);
    // s.pack("QpswiftInterface::settings::adaptive_rho", settings_.adaptive_rho);
    // s.pack("QpswiftInterface::settings::adaptive_rho_interval", settings_.adaptive_rho_interval);
    // s.pack("QpswiftInterface::settings::adaptive_rho_tolerance", settings_.adaptive_rho_tolerance);
    //s.pack("QpswiftInterface::settings::adaptive_rho_fraction", settings_.adaptive_rho_fraction);
    // s.pack("QpswiftInterface::settings::max_iter", settings_.max_iter);
    // s.pack("QpswiftInterface::settings::eps_abs", settings_.eps_abs);
    // s.pack("QpswiftInterface::settings::eps_rel", settings_.eps_rel);
    // s.pack("QpswiftInterface::settings::eps_prim_inf", settings_.eps_prim_inf);
    // s.pack("QpswiftInterface::settings::eps_dual_inf", settings_.eps_dual_inf);
    // s.pack("QpswiftInterface::settings::alpha", settings_.alpha);
    // s.pack("QpswiftInterface::settings::delta", settings_.delta);
    // s.pack("QpswiftInterface::settings::polish", settings_.polish);
    // s.pack("QpswiftInterface::settings::polish_refine_iter", settings_.polish_refine_iter);
    // s.pack("QpswiftInterface::settings::verbose", settings_.verbose);
    // s.pack("QpswiftInterface::settings::scaled_termination", settings_.scaled_termination);
    // s.pack("QpswiftInterface::settings::check_termination", settings_.check_termination);
    // s.pack("QpswiftInterface::settings::warm_start", settings_.warm_start);
    //s.pack("QpswiftInterface::settings::time_limit", settings_.time_limit);
  }

} // namespace casadi
