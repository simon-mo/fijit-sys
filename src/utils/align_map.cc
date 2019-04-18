#include "utils/align_map.h"

AlignmentDB& AlignmentDB::get_global_align_db() {
    static AlignmentDB db(path_);
    return db;
}