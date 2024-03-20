# use upcxx as the compiler

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}) # to include the autogen version file

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13 AND DEFINED UPCXX_LIBRARIES)
  message(STATUS "UPCXX_UTILS is using the UPCXX::upcxx library interface: ${UPCXX_LIBRARIES}")
else()
  find_program(UPCXX_EXEC upcxx)
  set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})
  message(STATUS "UPCXX_UTILS is using upcxx directly as the UPCXX::upcxx library interface is not available in this low version of cmake: ${CMAKE_VERSION}")
endif()

# construct individual template instances from header-file declared extern templates
# each can be compiled independently in a parallel build
set(EXTERN_TEMPLATE_FILES)
foreach(TEMPLATE_TYPE float double int uint64_t int64_t)
  set(TEMPLATE_FILE "timers-extern-template-${TEMPLATE_TYPE}")
  configure_file("timers-extern-template.in.cpp" "${TEMPLATE_FILE}.cpp" @ONLY)
  list(APPEND EXTERN_TEMPLATE_FILES ${TEMPLATE_FILE})
  foreach(TEMPLATE_OP "true" "false")
    set(TEMPLATE_TYPE2 "upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<${TEMPLATE_OP}> COMMA true>")
    set(TEMPLATE_FILE "reduce_prefix-extern-template-${TEMPLATE_TYPE}-min_not_max-${TEMPLATE_OP}")
    configure_file("reduce_prefix-extern-template.in.cpp" "${TEMPLATE_FILE}.cpp" @ONLY)
    list(APPEND EXTERN_TEMPLATE_FILES ${TEMPLATE_FILE})
  endforeach()
  foreach(TEMPLATE_OP "add")
    set(TEMPLATE_TYPE2 "upcxx::detail::op_wrap<upcxx::detail::opfn_${TEMPLATE_OP} COMMA true>")
    set(TEMPLATE_FILE "reduce_prefix-extern-template-${TEMPLATE_TYPE}-${TEMPLATE_OP}")
    configure_file("reduce_prefix-extern-template.in.cpp" "${TEMPLATE_FILE}.cpp" @ONLY)
    list(APPEND EXTERN_TEMPLATE_FILES ${TEMPLATE_FILE})
  endforeach()
endforeach()

set(upcxx_utils_libs log flat_aggr_store three_tier_aggr_store split_rank 
                     mem_profile ofstream binary_search reduce_prefix shared_array limit_outstanding promise_collectives
                     thread_pool 
                     ${EXTERN_TEMPLATE_FILES}
)

set(AND_THREADS)
if (THREADS_FOUND AND NOT UPCXX_UTILS_NO_THREADS)
  message(STATUS "Linking with Threads")
  set(AND_THREADS Threads::Threads)
endif()

list(APPEND upcxx_utils_libs_targets $<TARGET_OBJECTS:UPCXX_UTILS_VERSION>)

foreach(_lib ${upcxx_utils_libs})

  set(_libsname upcxx_utils_${_lib})
  add_library(${_libsname} OBJECT ${_lib}.cpp)
  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13 AND DEFINED UPCXX_LIBRARIES)
    target_link_libraries(${_libsname} PUBLIC ${UPCXX_LIBRARIES})
  endif()
  target_include_directories(${_libsname} INTERFACE
                       $<BUILD_INTERFACE:${UPCXX_UTILS_SOURCE_DIR}/include> # for headers when building
                       $<INSTALL_INTERFACE:include> # for client in install mode
  )

  # append to the list for aggreated library
  list(APPEND upcxx_utils_libs_targets $<TARGET_OBJECTS:${_libsname}>)

endforeach()

add_subdirectory(memory-allocators)

list(APPEND upcxx_utils_libs_targets $<TARGET_OBJECTS:UPCXX_UTILS_MEMORY_ALLOCATORS>)


# the aggregate library
add_library(UPCXX_UTILS_LIBRARY ${upcxx_utils_libs_targets})
set_property(TARGET UPCXX_UTILS_LIBRARY PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13 AND DEFINED UPCXX_LIBRARIES)
  target_link_libraries(UPCXX_UTILS_LIBRARY PUBLIC ${UPCXX_LIBRARIES} ${AND_THREADS})
endif()
target_include_directories(UPCXX_UTILS_LIBRARY INTERFACE
                       $<BUILD_INTERFACE:${UPCXX_UTILS_SOURCE_DIR}/include> # for headers when building
                       $<INSTALL_INTERFACE:include> # for client in install mode
)
install(TARGETS UPCXX_UTILS_LIBRARY EXPORT UPCXX_UTILS DESTINATION "lib")

