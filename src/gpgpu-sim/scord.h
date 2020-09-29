/********************************************************************************************
 * Copyright (c) 2020 Indian Institute of Science
 * All rights reserved.
 *
 * Developed by:    Aditya K Kamath, Alvin George A.
 *                  Computer Systems Lab
 *                  Indian Institute of Science
 *                  https://csl.csa.iisc.ac.in/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of Computer Systems Lab, Indian Institute of Science, 
 *        nor the names of its contributors may be used to endorse or promote products 
 *        derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ********************************************************************************************/
/* scord.h */
#ifndef SCORD_HEADER
#define SCORD_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <bitset>
#include <utility>
#include <set>
#include <queue>
#include <map>
#include <list>

class mem_fetch;
class mem_fetch_interface;
class gpgpu_t;
class memory_sub_partition;
class ptx_thread_info;
class warp_inst_t;
class OptionParser;

/*************************************************************************************/
// config
#define CFG_TRACKING_UNIT  VAL_TRACK_BY_WID
extern bool SCORD_ENABLED;
extern bool SCORD_PERF;
extern int GRANULARITY;
extern bool SCORD_RACE_EXIT;
extern bool SCORD_PRINT;
extern bool SCORD_SILENT;
#define printf_scord(...) if(!SCORD_SILENT && SCORD_PRINT) printf( __VA_ARGS__ )
void scord_reg_options(class OptionParser *  opp);

// (blkid (7) + syncid (8) + atomicid (16) + warpid (5) + modified (1) + {blk-}shared (1)
// + dev-shared (1) + devfenceid (6) + blkfenceid (6) + atom (1) + scope (1) + tag (4)) / 8
#define SCORD_SHADOW_SIZE 8
// (blkid(8) + warpid (8) + syncid (8) + atomicid (16)) / 8
#define SCORD_PACKET_SIZE 5

#define BLOOM_BINS 2
#define BLOOM_BIN_SIZE 8
#define LOCK_TABLE_SIZE 4

/*************************************************************************************/
void scord_kernel_restart(gpgpu_t *);

void scord_thread_genericfence(ptx_thread_info *pti, unsigned level);
void scord_thread_blkexecbarrier(ptx_thread_info *pti);

#define ATOMIC_LOCK ((unsigned)-1)
void scord_thread_atomgeneric(ptx_thread_info *pti, unsigned level, bool unlock, unsigned lockid);
void scord_locks_enable(ptx_thread_info *pti, unsigned level);

void scord_gmem_read(unsigned addr, unsigned size, ptx_thread_info *pti);
void scord_gmem_write(unsigned addr, unsigned size, ptx_thread_info *pti);

unsigned scord_execid(ptx_thread_info *thread);

/*************************************************************************************/

typedef unsigned fenceid_t;
typedef unsigned threadid_t;
typedef unsigned warpid_t;
typedef unsigned ctaid_t;
typedef unsigned addr_t;
typedef unsigned tag_t;
typedef std::pair<unsigned, unsigned> lockid_t;

class bloom_filter
{
public:
    bool empty()
    {
        bool flag = false;
        for(int i = 0; i < BLOOM_BINS; ++i)
        {
            if(filter[i].count() == 0)
                flag = true;
        }
        return locks.size() == 0 && flag;
    }

    void insert(lockid_t lock, bool enabled = false)
    {
        if(SCORD_PERF)
        {
            table t;
            getHash(lock, t.ind);
            t.valid = enabled;
            t.scope = (lock.second == 0 ? 0 : 1);

            for(std::list<table>::iterator it = locks.begin(); it != locks.end(); ++it)
            {
                bool flag = true;
                for(int j = 0; j < BLOOM_BINS; ++j)
                {
                    if(it->ind[j] != t.ind[j])
                    {
                        flag = false;
                        break;
                    }
                }

                if(flag && t.scope == it->scope)
                    return;
            }
            
            if(locks.size() == LOCK_TABLE_SIZE)
                locks.pop_front();

            locks.push_back(t);
        }
        else
        {
            if(func_locks.find(lock) == func_locks.end())
                func_locks[lock] = enabled;
        }
    }

    void enableLocks(unsigned level)
    {
        if(SCORD_PERF)
        {
            for(std::list<table>::iterator i = locks.begin(); i != locks.end(); ++i)
            {
                if(level == 407 || (level == 408 && i->scope != 0))
                    i->valid = true;
            }
        }
        else
        {
            for(std::map<lockid_t, bool>::iterator it = func_locks.begin(); it != func_locks.end(); ++it)
            {
                if(level == 407 || (level == 408 && it->first.second != 0))
                    it->second = true;
            }
        }
        
    }

    void erase(lockid_t lock)
    {
        clearFilter();
        if(SCORD_PERF)
        {
            table t;
            getHash(lock, t.ind);
            t.scope = (lock.second == 0 ? 0 : 1);

            for(std::list<table>::iterator it = locks.begin(); it != locks.end(); ++it)
            {
                bool flag = true;
                for(int j = 0; j < BLOOM_BINS; ++j)
                {
                    if(it->ind[j] != t.ind[j])
                    {
                        flag = false;
                        break;
                    }
                }

                if(flag && t.scope == it->scope)
                {
                    locks.erase(it);
                    return;
                }
            }
        }
        else
        {
            func_locks.erase(lock);
        }
        
    }

    unsigned long getVal()
    {
        setFilter();
        unsigned long val = 0;
        for(int i = 0; i < BLOOM_BINS; ++i)
        {
            val <<= BLOOM_BIN_SIZE;
            val += filter[i].to_ulong();
        }
        return val;
    }

    void setVal(unsigned long val)
    {
        clearFilter();
        for(int i = 0; i < BLOOM_BINS; ++i)
        {
            for(int j = 0; j < BLOOM_BIN_SIZE; ++j)
            {
                filter[BLOOM_BINS - i - 1][j] = ((val >> (i * BLOOM_BIN_SIZE + j)) & 1);
            }
        }
    }

    bloom_filter compare(bloom_filter b)
    {
        if(SCORD_PERF)
        {
            setFilter();
            b.setFilter();
            bloom_filter c;
            for(int i = 0; i < BLOOM_BINS; ++i)
                for(int j = 0; j < BLOOM_BIN_SIZE; ++j)
                    c.filter[i][j] = (filter[i][j] & b.filter[i][j]);
            return c;
        }
        else
        {
            bloom_filter c;
            for(std::map<lockid_t, bool>::iterator i = func_locks.begin(); i != func_locks.end(); ++i)
            {
                if(b.func_locks.find(i->first) != b.func_locks.end() && func_locks[i->first] && b.func_locks[i->first])
                    c.insert(i->first, true);
            }
            return c;
        }
    }

    bloom_filter() 
    {
        clearFilter();
    }
    ~bloom_filter() {}
private:
    struct table
    {
        int ind[BLOOM_BINS];
        bool valid;
        int scope;
    };
    std::list<table> locks;
    std::map<lockid_t, bool> func_locks;
    std::bitset<BLOOM_BIN_SIZE> filter[BLOOM_BINS];
    void clearFilter()
    {
        for(int i = 0; i < BLOOM_BINS; ++i)
            filter[i].reset();
    }
    void getHash(lockid_t lock, int *ind)
    {
        int val = (lock.first ^ lock.second);
        for(int j = 0; j < BLOOM_BINS; ++j)
        {
            ind[BLOOM_BINS - j - 1] = val & (BLOOM_BIN_SIZE - 1);
            val >>= 3;
        }
    }
    void setFilter()
    {
        if(SCORD_PERF)
        {
            for(std::list<table>::iterator it = locks.begin(); it != locks.end(); ++it)
            {
                if(!it->valid)
                    continue;
                
                for(int j = 0; j < BLOOM_BINS; ++j)
                {
                    filter[BLOOM_BINS - j - 1].set(it->ind[j]);
                }
            }
        }
        else
        {
            for(std::map<lockid_t, bool>::iterator i = func_locks.begin(); i != func_locks.end(); ++i)
            {
                if(!i->second)
                    continue;
                
                int val = (i->first.first ^ i->first.second);
                
                for(int j = 0; j < BLOOM_BINS; ++j)
                {
                    filter[BLOOM_BINS - j - 1].set(val & (BLOOM_BIN_SIZE - 1));
                    val >>= 3;
                }
            }
        }
    }
};
typedef bloom_filter lockset_t;
inline bloom_filter intersect_locks(bloom_filter &a, bloom_filter &b)
{
    return a.compare(b);
}

typedef struct scord_md_s {
    warpid_t   warpid;
    ctaid_t    ctaid;
    fenceid_t  blkfenceid;
    fenceid_t  devfenceid;
    bool       modified;
    bool       shared;
    lockset_t  lockset;
    fenceid_t  syncid;
    bool       devshared;
    bool       atom;
    bool       scope;
    tag_t      tag;
    bool       strong;
    scord_md_s();
} scord_metadata_t;

typedef struct scord_execdata_s {
    warpid_t   warpid;
    ctaid_t    ctaid;
    fenceid_t  blkfenceid;
    fenceid_t  devfenceid;
    lockset_t  lockset;
    fenceid_t  syncid;
    tag_t      tag;
    bool       atom;
    bool       scope;
    bool       vol;
    int        inst; // Used when reporting which instruction caused a race
    scord_execdata_s();
} scord_execdata_t;


void scord_create_packet_metadata(scord_execdata_t pti, mem_fetch *mf);
scord_execdata_t extract_packet_metadata(mem_fetch *mf);
void scord_get_execdata(warp_inst_t *inst);

#define STAT_PARAMS 32
class scord_detection_logic
{
public:
    void check_read_race(scord_md_s *md, scord_execdata_s *xd, fenceid_t blkfid, fenceid_t devfid);
    void check_write_race(scord_md_s *md, scord_execdata_s *xd, fenceid_t blkfid, fenceid_t devfid);
    scord_detection_logic() 
    {
        for(int i = 0; i < STAT_PARAMS; ++i)
            request_stats[i] = 0;
        races.clear();
        races2 = 0;
    }
    ~scord_detection_logic()
    {
        print_stats();
    }
    std::map<int, int> races;
    int races2;
    std::map<int, std::map<std::string, int> > race_types;
private:
    void make_owner(scord_md_s *md, scord_execdata_s *xd, bool write);
    void declare_race(char *msg, scord_metadata_t *md, scord_execdata_t *xd);
    int request_stats[STAT_PARAMS];
    void print_stats();    
    void update_stats(bool write, int m, int s, bool bid_match, bool locked)
    {
        int i = 0;
        i += (locked ? 1 : 0);
        i <<= 1;
        i += (bid_match ? 1 : 0);
        i <<= 1;
        i += (s ? 1 : 0);
        i <<= 1;
        i += (m ? 1 : 0);
        i <<= 1;
        i += (write ? 1 : 0);
        request_stats[i]++;
    }
};

class scord_request_generator
{
public:
    void reset(gpgpu_t *gpu);
    void generate_mem_request(mem_fetch **mf, memory_sub_partition **l2_cache, unsigned long long cycle);
    mem_fetch* process_mem_requests(memory_sub_partition **l2_cache, int line, unsigned long long cycle);
    bool busy() { return req_in_flight > 0; }
    scord_request_generator()
    {
        shadow_mem = NULL;
        shadow_size = 0;
        req_in_flight = 0;
        max_reqs = 0;
        cur_reqs = 0;
    }
    ~scord_request_generator()
    {
        printf("Max requests: %d\n", max_reqs);
    }
    addr_t shadow_mem;
    size_t shadow_size;
private:
    int req_in_flight;
    int max_reqs;
    int cur_reqs;
    std::map<unsigned, std::queue<std::pair<bool, scord_execdata_t> > > activeList;
    scord_metadata_t extract_shadow_metadata(unsigned char *);
    void scord_create_shadow_metadata(scord_metadata_t md, mem_fetch *mf, int addr);
    addr_t calculate_shadow_address(int addr);
};

extern scord_detection_logic gscord_detection;
extern scord_request_generator gscord_request;

#endif /* SCORD_HEADER */
